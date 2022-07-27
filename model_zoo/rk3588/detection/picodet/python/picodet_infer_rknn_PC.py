import argparse
import numpy as np
from utils.picodet_tool import PicodetPreProcess, hard_nms, softmax, warp_boxes, draw_box, label_list, sigmoid
from utils.rknn_config import RKNNConfigPC
import cv2

np.set_printoptions(suppress=True)
target = 416


# target = [416,416]

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--model_path',
        dest='model_path',
        default="../weights/onnx/picodet_xs_416_coco_lcnet_sim.onnx",
        type=str,
        help="path of model")
    parser.add_argument(
        '--export_model_path',
        dest='export_model_path',
        default="../weights/rknn/export.rknn",
        type=str,
        help="export path of model")
    parser.add_argument(
        '--need_export',
        dest='need_export',
        default=True,
        type=bool,
        help="export to rknn?")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold of score.")
    parser.add_argument(
        '--image_path',
        dest='image_path',
        default="../images/before/hrnet_demo.jpg",
        type=str,
        help='The directory or path or file list of the images to be predicted.')
    return parser.parse_args()


class PicodetRKNN:
    def __init__(self,
                 model_path,
                 target_size=None,
                 strides=None,
                 score_threshold=0.01,
                 nms_threshold=0.45,
                 nms_top_k=1000,
                 keep_top_k=100,
                 re_shape=target,
                 need_export=False,
                 export_path=None):
        self.input_shape = None
        if strides is None:
            self.strides = [8, 16, 32, 64]
        else:
            self.strides = strides
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.nms_top_k = nms_top_k
        self.keep_top_k = keep_top_k
        # Create RKNN object
        rknn_config = RKNNConfigPC(onnx_path=model_path, export_path=export_path, need_export=need_export)
        self.rknn = rknn_config.create_rknn()
        if target_size is None:
            self.target_size = [target, target]
        else:
            self.target_size = target_size
        self.re_shape = re_shape
        if need_export:
            self.need_export = need_export
        else:
            self.need_export = need_export
            self.export_path = export_path

    def infer(self, img):
        pic_pre_process = PicodetPreProcess(target_size=self.target_size)
        inputs, src_image = pic_pre_process.get_inputs(img)
        # print(inputs.shape)

        # print(inputs.shape)
        # print(inputs)
        result = self.rknn.inference([inputs])

        # for i in range(len(result)):
        #     print("result[{}].shape = {}".format(i, np.array(result[i]).shape))
        # print()
        result_out = []
        for i in range(2, len(result) // 2):
            # print("result[{}], result[{}] has appended".format(2 * i, 2 * i + 1))
            result_out.append(sigmoid(np.array(result[2 * i])) * sigmoid(np.array(result[2 * i + 1])))
        for i in range(len(result) // 4 + 1):
            # print("result[{}] has appended".format(i))
            result_out.append(result[i])
        # for i in range(len(result_out)):
            # print("result_out[{}].shape = {}".format(i, np.array(result_out[i]).shape))
        print()
        # print("result_out =",result_out)
        np_score_list = []
        np_boxes_list = []

        result = result_out
        num_outs = int(len(result) / 2)
        print("num_outs =", num_outs)
        for out_idx in range(num_outs):
            # print("result[out_idx].shape =", result[out_idx].shape)
            result[out_idx] = np.sqrt(result[out_idx])
            result[out_idx] = np.reshape(result[out_idx],
                                         newshape=(1, 80, result[out_idx].shape[2] * result[out_idx].shape[3]))
            result[out_idx] = np.transpose(result[out_idx], (0, 2, 1))
            np_score_list.append(result[out_idx])

            # print("result[out_idx + num_outs].shape =", result[out_idx + num_outs].shape)
            np_boxes_list.append(result[out_idx + num_outs])
        return np_score_list, np_boxes_list, inputs, src_image

    def detect(self, scores, raw_boxes):
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
                fm_h = self.input_shape[0] / stride
                fm_w = self.input_shape[1] / stride
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
                # print("topk_idx", topk_idx)
                # print("center.shape =", np.array(center).shape)
                center = center[topk_idx]
                # print("center.shape =", np.array(center).shape)
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
        np_score_list, np_boxes_list, image, src_image = self.infer(img)
        self.input_shape = image.shape[1:3]
        # print("src_img.shape =", src_img.shape)
        # print("self.input_shape =", self.input_shape)

        # detect
        out_boxes_list, out_boxes_num = self.detect(np_score_list, np_boxes_list)

        scale_x = src_image.shape[1] / image.shape[2]
        scale_y = src_image.shape[0] / image.shape[1]
        res_image = draw_box(src_image, out_boxes_list, label_list, scale_x, scale_y)
        #
        cv2.imwrite('../images/after/rknn_PC_result.jpg', res_image)


if __name__ == '__main__':
    FLAGS = parse_args()
    model_path = FLAGS.model_path
    imgs_path = FLAGS.image_path
    need_export = FLAGS.need_export
    export_model_path = FLAGS.export_model_path
    picodet_infer_onnx = PicodetRKNN(model_path=model_path, need_export=need_export, export_path=export_model_path)
    picodet_infer_onnx.predict(imgs_path)
