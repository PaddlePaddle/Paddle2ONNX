# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import yaml
import argparse

import numpy as np
from functools import reduce

from utils.visualize import save_imgs
from utils.preprocess import preprocess, Resize, NormalizeImage, Permute, PadStride


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--model_path',
        dest='model_path',
        type=str,
        help="while use_paddle_predict, this means directory path of paddle model. Other wise, this means path of "
        "onnx model file. and include 'infer_cfg.yml'")
    parser.add_argument(
        '--image_path',
        dest='image_path',
        type=str,
        help='The directory or path or file list of the images to be predicted.')
    parser.add_argument(
        '--use_paddle_predict',
        type=bool,
        default=False,
        help="If use paddlepaddle to predict, otherwise use onnxruntime to predict."
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5, help="Threshold of score.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Directory of output visualization files.")
    return parser.parse_args()


class PredictConfig():
    def __init__(self, model_path):
        # parsing Yaml config for Preprocess
        deploy_file = os.path.join(model_path, 'infer_cfg.yml')
        with open(deploy_file) as f:
            yml_conf = yaml.safe_load(f)
        self.arch = yml_conf['arch']
        self.preprocess_infos = yml_conf['Preprocess']
        self.labels = yml_conf['label_list']


class DetPreProcess(object):
    def __init__(self, pred_config):
        self.pred_config = pred_config

    def __call__(self, im):
        preprocess_ops = []
        for op_info in self.pred_config.preprocess_infos:
            new_op_info = op_info.copy()
            op_type = new_op_info.pop('type')
            preprocess_ops.append(eval(op_type)(**new_op_info))

        input_im_lst = []
        input_im_info_lst = []
        im, im_info = preprocess(im, preprocess_ops)
        input_im_lst.append(im)
        input_im_info_lst.append(im_info)
        inputs = {}
        inputs['image'] = np.array((im, )).astype('float32')
        if self.pred_config.arch == 'YOLO':
            inputs['im_shape'] = np.array(
                (im_info['im_shape'], )).astype('float32')
        inputs['scale_factor'] = np.array(
            (im_info['scale_factor'], )).astype('float32')
        return inputs


class DetPostProcess(object):
    def __init__(self,
                 score_threshold=0.01,
                 nms_threshold=0.45,
                 nms_top_k=1000,
                 keep_top_k=100):
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.nms_top_k = nms_top_k
        self.keep_top_k = keep_top_k

    # 计算IoU，矩形框的坐标形式为xyxy
    def box_iou_xyxy(self, box1, box2):
        # 获取box1左上角和右下角的坐标
        x1min, y1min, x1max, y1max = box1[0], box1[1], box1[2], box1[3]
        # 计算box1的面积
        s1 = (y1max - y1min + 1.) * (x1max - x1min + 1.)
        # 获取box2左上角和右下角的坐标
        x2min, y2min, x2max, y2max = box2[0], box2[1], box2[2], box2[3]
        # 计算box2的面积
        s2 = (y2max - y2min + 1.) * (x2max - x2min + 1.)

        # 计算相交矩形框的坐标
        xmin = np.maximum(x1min, x2min)
        ymin = np.maximum(y1min, y2min)
        xmax = np.minimum(x1max, x2max)
        ymax = np.minimum(y1max, y2max)
        # 计算相交矩形行的高度、宽度、面积
        inter_h = np.maximum(ymax - ymin + 1., 0.)
        inter_w = np.maximum(xmax - xmin + 1., 0.)
        intersection = inter_h * inter_w
        # 计算相并面积
        union = s1 + s2 - intersection
        # 计算交并比
        iou = intersection / union
        return iou

    # 非极大值抑制
    def nms(self,
            bboxes,
            scores,
            score_thresh,
            nms_thresh,
            pre_nms_topk,
            i=0,
            c=0):
        """
        nms
        """
        inds = np.argsort(scores)
        inds = inds[::-1]
        keep_inds = []
        while (len(inds) > 0):
            cur_ind = inds[0]
            cur_score = scores[cur_ind]
            # if score of the box is less than score_thresh, just drop it
            if cur_score < score_thresh:
                break

            keep = True
            for ind in keep_inds:
                current_box = bboxes[cur_ind]
                remain_box = bboxes[ind]
                iou = self.box_iou_xyxy(current_box, remain_box)
                if iou > nms_thresh:
                    keep = False
                    break
            if i == 0 and c == 4 and cur_ind == 951:
                print('suppressed, ', keep, i, c, cur_ind, ind, iou)
            if keep:
                keep_inds.append(cur_ind)
            inds = inds[1:]

        return np.array(keep_inds)

    # 多分类非极大值抑制
    def multiclass_nms(self,
                       bboxes,
                       scores,
                       score_threshold=0.01,
                       nms_threshold=0.45,
                       nms_top_k=1000,
                       keep_top_k=100):
        """
        This is for multiclass_nms
        """
        batch_size = bboxes.shape[0]
        class_num = scores.shape[1]
        rets = []
        for i in range(batch_size):
            bboxes_i = bboxes[i]
            scores_i = scores[i]
            ret = []
            for c in range(class_num):
                scores_i_c = scores_i[c]
                keep_inds = self.nms(bboxes_i,
                                     scores_i_c,
                                     score_threshold,
                                     nms_threshold,
                                     nms_top_k,
                                     i=i,
                                     c=c)
                if len(keep_inds) < 1:
                    continue
                keep_bboxes = bboxes_i[keep_inds]
                keep_scores = scores_i_c[keep_inds]
                keep_results = np.zeros([keep_scores.shape[0], 6])
                keep_results[:, 0] = c
                keep_results[:, 1] = keep_scores[:]
                keep_results[:, 2:6] = keep_bboxes[:]
                ret.append(keep_results)
            if len(ret) < 1:
                rets.append(ret)
                continue
            ret_i = np.concatenate(ret, axis=0)
            scores_i = ret_i[:, 1]
            if len(scores_i) > keep_top_k:
                inds = np.argsort(scores_i)[::-1]
                inds = inds[:keep_top_k]
                ret_i = ret_i[inds]

            rets.append(ret_i)
        return rets

    def __call__(self, bboxes, scores):
        # nms
        bbox_pred = self.multiclass_nms(
            bboxes,
            scores,
            score_threshold=self.score_threshold,
            nms_threshold=self.nms_threshold,
            nms_top_k=self.nms_top_k,
            keep_top_k=self.keep_top_k)
        np_boxes = bbox_pred[0]
        np_boxes_num = [bbox_pred[0].shape[0]]
        if reduce(lambda x, y: x * y, np_boxes.shape) < 6:
            print('[WARNNING] No object detected.')
            np_boxes = np.zeros([0, 6])
            np_boxes_num = [0]
        results = dict(boxes=np_boxes, boxes_num=np_boxes_num)
        return results


def onnx_predict(model_path, imgs_path, pred_config):
    import onnxruntime as rt
    sess = rt.InferenceSession(os.path.join(model_path, 'model.onnx'))

    # preprocess
    preProcess = DetPreProcess(pred_config)
    inputs = preProcess(imgs_path)
    output = sess.run(None, inputs)

    # postprocess
    postProcess = DetPostProcess(score_threshold=0.01, nms_threshold=0.45)
    results = postProcess(output[0], output[1])
    return results


def paddle_predict(model_path, imgs_path, pred_config):
    import paddle
    model = paddle.jit.load(os.path.join(model_path, 'model'))
    model.eval()

    # preprocess
    preProcess = DetPreProcess(pred_config)
    inputs = preProcess(imgs_path)
    if pred_config.arch == "YOLO":
        output = model(inputs['im_shape'], inputs['image'],
                       inputs['scale_factor'])
    else:
        output = model(inputs['image'], inputs['scale_factor'])

    # postprocess
    postProcess = DetPostProcess(score_threshold=0.01, nms_threshold=0.45)
    results = postProcess(output[0].numpy(), output[1].numpy())
    return results


if __name__ == '__main__':
    FLAGS = parse_args()
    imgs_path = FLAGS.image_path
    pred_config = PredictConfig(FLAGS.model_path)

    if FLAGS.use_paddle_predict:
        paddle_result = paddle_predict(FLAGS.model_path, imgs_path, pred_config)
        save_imgs(
            paddle_result,
            imgs_path,
            labels=pred_config.labels,
            output_dir=FLAGS.output_dir,
            threshold=FLAGS.threshold,
            prefix="paddle")
    else:
        onnx_result = onnx_predict(FLAGS.model_path, imgs_path, pred_config)
        save_imgs(
            onnx_result,
            imgs_path,
            labels=pred_config.labels,
            output_dir=FLAGS.output_dir,
            threshold=FLAGS.threshold,
            prefix="onnx")
