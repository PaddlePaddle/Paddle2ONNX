import argparse
from utils.picodet_tool import PicodetPreProcess, hard_nms, softmax, warp_boxes, draw_box \
    , label_list, get_coco_label_id, coco_label_id
import numpy as np
import cv2
import os
from pathlib import Path
import json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import time


def str2bool(v):
    return v.lower() in ("true", "t", "1")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--model_path',
        dest='model_path',
        default="./weights/onnx/picodet_s_320_coco_sim.onnx",
        type=str,
        help="path of model")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold of score.")
    parser.add_argument(
        '--image_path',
        dest='image_path',
        default="./images/before/000000000139.jpg",
        type=str,
        help='The directory or path or file list of the images to be predicted.')
    parser.add_argument(
        '--backend_type',
        dest='backend_type',
        help='The type for reading the model.',
        type=str,
        choices=["rk_board", "rk_pc", "onnx"],
        default='onnx')
    parser.add_argument(
        '--save_path',
        dest='save_path',
        help='The file for saving the predict result.',
        type=str,
        default='./images/after/000000000139.jpg')
    parser.add_argument(
        '--do_quantization',
        dest='do_quantization',
        help='do quantization?',
        type=str2bool,
        default=False)
    parser.add_argument(
        '--quantization_list',
        dest='quantization_list',
        help='list file path for do quantization',
        type=str,
        default=None)
    parser.add_argument(
        '--target',
        dest='target',
        help='The target for build rknn.',
        type=str,
        default='RK3588')
    return parser.parse_args()


class Picodet:
    def __init__(self,
                 model_path,
                 target_size=None,
                 strides=None,
                 score_threshold=0.01,
                 nms_threshold=0.5,
                 nms_top_k=1000,
                 keep_top_k=100,
                 re_shape=320,
                 backend_type="onnx",
                 target="RK3588"):
        if strides is None:
            strides = [8, 16, 32, 64]
        if target_size is None:
            target_size = [320, 320]
        self.model_path = model_path
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.nms_top_k = nms_top_k
        self.keep_top_k = keep_top_k
        self.re_shape = re_shape
        self.strides = strides
        self.target_size = target_size
        self.backend_type = backend_type
        self.target = target
        if self.backend_type == "onnx":
            from utils.ONNXConfig import ONNXConfig
            self.model = ONNXConfig(self.model_path)
        elif self.backend_type == "rk_pc":
            from utils.RKNNConfig import RKNNConfigPC
            rknn_config = RKNNConfigPC(self.model_path, self.target)
            if not FLAGS.do_quantization:
                self.model = rknn_config.create_rknn(verbose=False)
            # self.model = rknn_config.create_rknn_hybrid_quantization_step1()
            else:
                self.model = rknn_config.create_rknn_hybrid_quantization_step2()
        elif self.backend_type == "rk_board":
            from utils.RKNNConfig import RKNNConfigBoard
            self.model = RKNNConfigBoard(self.model_path, self.target).create_rknn(verbose=False)

    def infer_by_onnx(self, img):
        pic_pre_process = PicodetPreProcess(target_size=self.target_size)
        inputs, src_image = pic_pre_process.get_inputs(img)
        inputs = np.array((inputs,)).astype('float32')
        start_time = time.time()
        result = self.model.infer(inputs)
        end_time = time.time()
        return result, inputs, src_image, end_time - start_time

    def infer_by_rknn_pc(self, img, verbose=True):
        pic_pre_process = PicodetPreProcess(target_size=self.target_size)
        inputs, src_image = pic_pre_process.get_inputs(img)
        new_inputs = inputs.transpose((1, 2, 0))  # chw hwc
        new_inputs = np.array((new_inputs,)).astype('float32')
        inputs = np.array((inputs,))
        start_time = time.time()
        result = self.model.inference([new_inputs])
        end_time = time.time()
        return result, inputs, src_image, end_time - start_time

    def infer_by_rknn_board(self, img, verbose=True):
        pic_pre_process = PicodetPreProcess(target_size=self.target_size)
        inputs, src_image = pic_pre_process.get_inputs(img)
        new_inputs = inputs.transpose((1, 2, 0))  # chw hwc
        new_inputs = np.array((new_inputs,)).astype('float32')
        inputs = np.array((inputs,))
        start_time = time.time()
        result = self.model.inference([new_inputs])
        end_time = time.time()
        for i in range(len(result)):
            result[i] = np.resize(result[i], (result[i].shape[0:3]))
        return result, inputs, src_image, end_time - start_time

    def detect(self, scores, raw_boxes, input_shape):
        # detect
        test_im_shape = np.array([[self.re_shape, self.re_shape]]).astype('float32')
        test_scale_factor = np.array([[1, 1]]).astype('float32')
        batch_size = raw_boxes[0].shape[0]
        reg_max = int(raw_boxes[0].shape[-1] / 4 - 1)
        out_boxes_num = []
        out_boxes_list = []
        for batch_id in range(batch_size):
            # generate centers
            decode_boxes = []
            select_scores = []
            for stride, box_distribute, score in zip(self.strides, raw_boxes, scores):
                box_distribute = box_distribute[batch_id]
                score = score[batch_id]
                # centers
                fm_h = input_shape[0] / stride
                fm_w = input_shape[1] / stride
                # print("fm_h = {},fm_w = {}".format(fm_h, fm_w))
                h_range = np.arange(fm_h)
                w_range = np.arange(fm_w)
                ww, hh = np.meshgrid(w_range, h_range)
                ct_row = (hh.flatten() + 0.5) * stride
                ct_col = (ww.flatten() + 0.5) * stride
                # print("ct_row.shape = {},ct_col.shape = {}".format(ct_row.shape, ct_col.shape))
                center = np.stack((ct_col, ct_row, ct_col, ct_row), axis=1)

                # box distribution to distance
                reg_range = np.arange(reg_max + 1)
                box_distance = box_distribute.reshape((-1, reg_max + 1))
                # print("softmax shape =", box_distance.shape)
                box_distance = softmax(box_distance)
                # print("softmax shape =", box_distance.shape)
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
                box_probs = hard_nms(
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
                picked_box_probs[:, :4] = warp_boxes(
                    picked_box_probs[:, :4], test_im_shape[batch_id])
                im_scale = np.concatenate([
                    test_scale_factor[batch_id][::-1],
                    test_scale_factor[batch_id][::-1]
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

    def predict(self, img):
        if self.backend_type == "onnx":
            result, image, src_image, _ = self.infer_by_onnx(img)
        elif self.backend_type == "rk_pc":
            result, image, src_image, _ = self.infer_by_rknn_pc(img, verbose=False)
        elif self.backend_type == "rk_board":
            result, image, src_image, _ = self.infer_by_rknn_board(img, verbose=False)
        # for i in result:
        #     print(i.shape)
        np_score_list = []
        np_boxes_list = []
        num_outs = int(len(result) / 2)
        for out_idx in range(num_outs):
            np_score_list.append(result[out_idx])
            np_boxes_list.append(result[out_idx + num_outs])

        out_boxes_list, out_boxes_num = self.detect(np_score_list, np_boxes_list, self.target_size)

        scale_x = src_image.shape[1] / image.shape[3]
        scale_y = src_image.shape[0] / image.shape[2]

        res_image = draw_box(src_image, out_boxes_list, label_list, scale_x, scale_y)
        t_ls = os.path.basename(FLAGS.save_path).split(".")
        save_path = os.path.dirname(FLAGS.save_path) + "/" + t_ls[0] + "_" + self.backend_type + "." + t_ls[-1]
        cv2.imwrite(save_path, res_image)

    def eval(self, img, img_num):
        self.score_threshold = 0.01
        pred_results = []
        img_paths = os.listdir(img)[0:img_num]
        # print(img_paths)
        count = 0
        all_time = 0
        for t_img in img_paths:
            count += 1
            print(count)
            input_img = os.path.join(img, t_img)
            # print(input_img)
            if self.backend_type == "onnx":
                result, image, src_image, t_time = self.infer_by_onnx(input_img)
            elif self.backend_type == "rk_pc":
                result, image, src_image, t_time = self.infer_by_rknn_pc(input_img, verbose=False)
            elif self.backend_type == "rk_board":
                result, image, src_image, t_time = self.infer_by_rknn_board(input_img, verbose=False)
            all_time += t_time
            np_score_list = []
            np_boxes_list = []
            num_outs = int(len(result) / 2)
            for out_idx in range(num_outs):
                np_score_list.append(result[out_idx])
                np_boxes_list.append(result[out_idx + num_outs])

            out_boxes_list, out_boxes_num = self.detect(np_score_list, np_boxes_list, self.target_size)

            scale_x = src_image.shape[1] / image.shape[3]
            scale_y = src_image.shape[0] / image.shape[2]

            path = int(Path(input_img).stem)
            pred_results.extend(convert_to_coco_format(path, out_boxes_list, label_list, scale_x, scale_y))
        # print(pred_results)
        pred_json = "./predictions.json"
        with open(pred_json, 'w') as f:
            json.dump(pred_results, f)
        anno = COCO("./images/coco/annotations/instances_val2017.json")
        pred = anno.loadRes(pred_json)
        cocoEval = COCOeval(anno, pred, 'bbox')
        # print(get_ids(img_paths))
        cocoEval.params.imgIds = get_ids(img_paths)
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        print("time = ", round(all_time / img_num, 3))


def convert_to_coco_format(img_id, results, label_list, scale_x, scale_y):
    pred_results = []
    for i in range(len(results)):
        bbox = results[i, 2:]
        label_id = int(results[i, 0])
        score = round(results[i, 1], 5)
        bbox = [round(bbox[0] * scale_x, 3),
                round(bbox[1] * scale_y, 3),
                round(bbox[2] * scale_x, 3) - round(bbox[0] * scale_x, 3),
                round(bbox[3] * scale_y, 3) - round(bbox[1] * scale_y, 3)]
        pred_data = {
            "image_id": img_id,
            "category_id": coco_label_id[label_list[label_id]],
            "bbox": bbox,
            "score": score
        }
        pred_results.append(pred_data)
    return pred_results


def get_ids(imgs):
    ids = []
    for img in imgs:
        # print(img)
        ids.append(int(Path(img).stem))
    return ids


def create_dataset(img, img_num, save_path):
    img_paths = os.listdir(img)[0:img_num]
    with open(os.path.join(save_path, "dataset.txt"), "w") as f:
        for img_path in img_paths:
            f.write(os.path.join("./images/", img_path) + "\n")


if __name__ == "__main__":
    FLAGS = parse_args()
    picodet = Picodet(model_path=FLAGS.model_path,
                      backend_type=FLAGS.backend_type,
                      score_threshold=FLAGS.threshold)
    picodet.predict(FLAGS.image_path)

    # eval
    # get_coco_label_id()
    # create_dataset("./images/coco/images", 100, "./images/coco")
    # picodet.eval("./images/coco/images", 700)
