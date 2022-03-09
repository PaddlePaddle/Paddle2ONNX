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

import cv2
import argparse

import numpy as np
from scipy.special import softmax

from utils.visualize import save_imgs


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--model_path',
        dest='model_path',
        type=str,
        help="while use_paddle_predict, this means directory path of paddle model. Other wise, this means path of "
        "onnx model file.")
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
        default="output",
        help="Directory of output visualization files.")
    return parser.parse_args()


class PicoDetPreProcess(object):
    def __init__(self, ):
        self.target_size = [640, 640]
        self.interp = cv2.INTER_CUBIC
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.is_scale = True
        self.stride = 32

    def resize(self, im, im_info):
        target_size = self.target_size
        interp = self.interp
        assert len(target_size) == 2
        assert target_size[0] > 0 and target_size[1] > 0
        origin_shape = im.shape[:2]
        resize_h, resize_w = target_size
        im_scale_y = resize_h / float(origin_shape[0])
        im_scale_x = resize_w / float(origin_shape[1])
        im = cv2.resize(
            im, None, None, fx=im_scale_x, fy=im_scale_y, interpolation=interp)
        im_info['im_shape'] = np.array(im.shape[:2]).astype('float32')
        im_info['scale_factor'] = np.array(
            [im_scale_y, im_scale_x]).astype('float32')
        return im, im_info

    def normalizeImage(self, im):
        mean = self.mean
        std = self.std
        is_scale = self.is_scale

        im = im.astype(np.float32, copy=False)
        mean = np.array(mean)[np.newaxis, np.newaxis, :]
        std = np.array(std)[np.newaxis, np.newaxis, :]

        if is_scale:
            im = im / 255.0
        im -= mean
        im /= std
        return im

    def padStride(self, im):
        coarsest_stride = self.stride
        coarsest_stride = coarsest_stride
        if coarsest_stride <= 0:
            return im
        im_c, im_h, im_w = im.shape
        pad_h = int(np.ceil(float(im_h) / coarsest_stride) * coarsest_stride)
        pad_w = int(np.ceil(float(im_w) / coarsest_stride) * coarsest_stride)
        padding_im = np.zeros((im_c, pad_h, pad_w), dtype=np.float32)
        padding_im[:, :im_h, :im_w] = im
        return padding_im

    def __call__(self, im):
        im_info = {
            'scale_factor': np.array(
                [1., 1.], dtype=np.float32),
            'im_shape': None,
        }
        im = cv2.imread(im)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im, im_info = self.resize(im, im_info)
        im = self.normalizeImage(im)
        im = im.transpose((2, 0, 1))
        im = self.padStride(im)

        inputs = {}
        inputs['image'] = np.array((im, )).astype('float32')
        inputs['im_shape'] = np.array((im_info['im_shape'], )).astype('float32')
        inputs['scale_factor'] = np.array(
            (im_info['scale_factor'], )).astype('float32')

        return inputs


class PicoDetPostProcess(object):
    """
    Args:
        input_shape (int): network input image size
        ori_shape (int): ori image shape of before padding
        scale_factor (float): scale factor of ori image
        enable_mkldnn (bool): whether to open MKLDNN
    """

    def __init__(self,
                 input_shape,
                 ori_shape,
                 scale_factor,
                 strides=[8, 16, 32, 64],
                 score_threshold=0.4,
                 nms_threshold=0.5,
                 nms_top_k=1000,
                 keep_top_k=100):
        self.ori_shape = ori_shape
        self.input_shape = input_shape
        self.scale_factor = scale_factor
        self.strides = strides
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.nms_top_k = nms_top_k
        self.keep_top_k = keep_top_k

    def hard_nms(self, box_scores, iou_threshold, top_k=-1, candidate_size=200):
        """
        Args:
            box_scores (N, 5): boxes in corner-form and probabilities.
            iou_threshold: intersection over union threshold.
            top_k: keep top_k results. If k <= 0, keep all the results.
            candidate_size: only consider the candidates with the highest scores.
        Returns:
             picked: a list of indexes of the kept boxes
        """
        scores = box_scores[:, -1]
        boxes = box_scores[:, :-1]
        picked = []
        indexes = np.argsort(scores)
        indexes = indexes[-candidate_size:]
        while len(indexes) > 0:
            current = indexes[-1]
            picked.append(current)
            if 0 < top_k == len(picked) or len(indexes) == 1:
                break
            current_box = boxes[current, :]
            indexes = indexes[:-1]
            rest_boxes = boxes[indexes, :]
            iou = self.iou_of(
                rest_boxes,
                np.expand_dims(
                    current_box, axis=0), )
            indexes = indexes[iou <= iou_threshold]

        return box_scores[picked, :]

    def iou_of(self, boxes0, boxes1, eps=1e-5):
        """Return intersection-over-union (Jaccard index) of boxes.
        Args:
            boxes0 (N, 4): ground truth boxes.
            boxes1 (N or 1, 4): predicted boxes.
            eps: a small number to avoid 0 as denominator.
        Returns:
            iou (N): IoU values.
        """
        overlap_left_top = np.maximum(boxes0[..., :2], boxes1[..., :2])
        overlap_right_bottom = np.minimum(boxes0[..., 2:], boxes1[..., 2:])

        overlap_area = self.area_of(overlap_left_top, overlap_right_bottom)
        area0 = self.area_of(boxes0[..., :2], boxes0[..., 2:])
        area1 = self.area_of(boxes1[..., :2], boxes1[..., 2:])
        return overlap_area / (area0 + area1 - overlap_area + eps)

    def area_of(self, left_top, right_bottom):
        """Compute the areas of rectangles given two corners.
        Args:
            left_top (N, 2): left top corner.
            right_bottom (N, 2): right bottom corner.
        Returns:
            area (N): return the area.
        """
        hw = np.clip(right_bottom - left_top, 0.0, None)
        return hw[..., 0] * hw[..., 1]

    def warp_boxes(self, boxes, ori_shape):
        """Apply transform to boxes
        """
        width, height = ori_shape[1], ori_shape[0]
        n = len(boxes)
        if n:
            # warp points
            xy = np.ones((n * 4, 3))
            xy[:, :2] = boxes[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(
                n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
            # xy = xy @ M.T  # transform
            xy = (xy[:, :2] / xy[:, 2:3]).reshape(n, 8)  # rescale
            # create new boxes
            x = xy[:, [0, 2, 4, 6]]
            y = xy[:, [1, 3, 5, 7]]
            xy = np.concatenate(
                (x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T
            # clip boxes
            xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, width)
            xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, height)
            return xy.astype(np.float32)
        else:
            return boxes

    def __call__(self, scores, raw_boxes):
        batch_size = raw_boxes[0].shape[0]
        reg_max = int(raw_boxes[0].shape[-1] / 4 - 1)
        out_boxes_num = []
        out_boxes_list = []
        for batch_id in range(batch_size):
            # generate centers
            decode_boxes = []
            select_scores = []
            for stride, box_distribute, score in zip(self.strides, raw_boxes,
                                                     scores):
                box_distribute = box_distribute[batch_id]
                score = score[batch_id]
                # centers
                fm_h = self.input_shape[0] / stride
                fm_w = self.input_shape[1] / stride
                h_range = np.arange(fm_h)
                w_range = np.arange(fm_w)
                ww, hh = np.meshgrid(w_range, h_range)
                ct_row = (hh.flatten() + 0.5) * stride
                ct_col = (ww.flatten() + 0.5) * stride
                center = np.stack((ct_col, ct_row, ct_col, ct_row), axis=1)

                # box distribution to distance
                reg_range = np.arange(reg_max + 1)
                box_distance = box_distribute.reshape((-1, reg_max + 1))
                box_distance = softmax(box_distance, axis=1)
                box_distance = box_distance * np.expand_dims(reg_range, axis=0)
                box_distance = np.sum(box_distance, axis=1).reshape((-1, 4))
                box_distance = box_distance * stride

                # top K candidate
                topk_idx = np.argsort(score.max(axis=1))[::-1]
                topk_idx = topk_idx[:self.nms_top_k]
                center = center[topk_idx]
                score = score[topk_idx]
                box_distance = box_distance[topk_idx]

                # decode box
                decode_box = center + [-1, -1, 1, 1] * box_distance

                select_scores.append(score)
                decode_boxes.append(decode_box)

            # nms
            bboxes = np.concatenate(decode_boxes, axis=0)
            confidences = np.concatenate(select_scores, axis=0)
            picked_box_probs = []
            picked_labels = []
            for class_index in range(0, confidences.shape[1]):
                probs = confidences[:, class_index]
                mask = probs > self.score_threshold
                probs = probs[mask]
                if probs.shape[0] == 0:
                    continue
                subset_boxes = bboxes[mask, :]
                box_probs = np.concatenate(
                    [subset_boxes, probs.reshape(-1, 1)], axis=1)
                box_probs = self.hard_nms(
                    box_probs,
                    iou_threshold=self.nms_threshold,
                    top_k=self.keep_top_k, )
                picked_box_probs.append(box_probs)
                picked_labels.extend([class_index] * box_probs.shape[0])

            if len(picked_box_probs) == 0:
                out_boxes_list.append(np.empty((0, 4)))
                out_boxes_num.append(0)

            else:
                picked_box_probs = np.concatenate(picked_box_probs)

                # resize output boxes
                picked_box_probs[:, :4] = self.warp_boxes(
                    picked_box_probs[:, :4], self.ori_shape[batch_id])
                im_scale = np.concatenate([
                    self.scale_factor[batch_id][::-1],
                    self.scale_factor[batch_id][::-1]
                ])
                picked_box_probs[:, :4] /= im_scale
                # clas score box
                out_boxes_list.append(
                    np.concatenate(
                        [
                            np.expand_dims(
                                np.array(picked_labels),
                                axis=-1), np.expand_dims(
                                    picked_box_probs[:, 4], axis=-1),
                            picked_box_probs[:, :4]
                        ],
                        axis=1))
                out_boxes_num.append(len(picked_labels))

        out_boxes_list = np.concatenate(out_boxes_list, axis=0)
        out_boxes_num = np.asarray(out_boxes_num).astype(np.int32)
        return out_boxes_list, out_boxes_num


def onnx_predict(onnx_path, imgs_path):
    import onnxruntime as rt
    sess = rt.InferenceSession(onnx_path)
    # preprocess
    preProcess = PicoDetPreProcess()
    inputs = preProcess(imgs_path)
    onxx_input_list = {}
    for i in range(len(sess.get_inputs())):
        input_name = sess.get_inputs()[i].name
        onxx_input_list[input_name] = inputs[input_name]
    output = sess.run(None, onxx_input_list)

    # model_prediction
    np_score_list, np_boxes_list = [], []
    num_outs = 4
    for out_idx in range(num_outs):
        scores = output[out_idx]
        np_score_list.append(scores)
        boxes = output[out_idx + num_outs]
        np_boxes_list.append(boxes)

    # postprocess
    postProcess = PicoDetPostProcess(
        inputs['image'].shape[2:],
        inputs['im_shape'],
        inputs['scale_factor'],
        strides=[8, 16, 32, 64],
        nms_threshold=0.5)
    np_boxes, np_boxes_num = postProcess(np_score_list, np_boxes_list)
    results = dict(boxes=np_boxes, boxes_num=np_boxes_num)
    return results


def paddle_predict(model_path, imgs_path):
    import paddle
    model = paddle.jit.load(model_path)
    model.eval()

    # preprocess
    preProcess = PicoDetPreProcess()
    inputs = preProcess(imgs_path)
    output = model(inputs['image'])

    # model_prediction
    np_score_list, np_boxes_list = [], []
    num_outs = 4
    for out_idx in range(num_outs):
        scores = output[out_idx].numpy()
        np_score_list.append(scores)
        boxes = output[out_idx + num_outs].numpy()
        np_boxes_list.append(boxes)

    # postprocess
    postProcess = PicoDetPostProcess(
        inputs['image'].shape[2:],
        inputs['im_shape'],
        inputs['scale_factor'],
        strides=[8, 16, 32, 64],
        nms_threshold=0.5)
    np_boxes, np_boxes_num = postProcess(np_score_list, np_boxes_list)
    results = dict(boxes=np_boxes, boxes_num=np_boxes_num)
    return results


if __name__ == '__main__':
    FLAGS = parse_args()
    imgs_path = FLAGS.image_path

    if FLAGS.use_paddle_predict:
        paddle_result = paddle_predict(FLAGS.model_path, imgs_path)
        save_imgs(
            paddle_result,
            imgs_path,
            output_dir=FLAGS.output_dir,
            threshold=FLAGS.threshold,
            prefix="paddle")
    else:
        onnx_result = onnx_predict(FLAGS.model_path, imgs_path)
        save_imgs(
            onnx_result,
            imgs_path,
            output_dir=FLAGS.output_dir,
            threshold=FLAGS.threshold,
            prefix="onnx")
