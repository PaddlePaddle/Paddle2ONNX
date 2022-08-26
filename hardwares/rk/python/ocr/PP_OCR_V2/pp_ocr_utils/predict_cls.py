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
import sys
sys.path.append('../../../')
import cv2
import copy
import numpy as np
import math


# from sklearn.utils.extmath import softmax


def softmax(x):
    """ softmax function """

    # assert(len(x.shape) > 1, "dimension must be larger than 1")
    # print(np.max(x, axis = 1, keepdims = True)) # axis = 1, 行

    x -= np.max(x, axis=1, keepdims=True)  # 为了稳定地计算softmax概率， 一般会减掉最大的那个元素

    print("减去行最大值 ：\n", x)

    x = np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

    return x


class ClsPostProcess(object):
    """ Convert between text-label and text-index """

    def __init__(self, label_list, **kwargs):
        super(ClsPostProcess, self).__init__()
        self.label_list = label_list

    def __call__(self, preds, label=None, *args, **kwargs):
        # if isinstance(preds, paddle.Tensor):
        #     preds = preds.numpy()
        pred_idxs = preds.argmax(axis=1)
        decode_out = [(self.label_list[idx], preds[i, idx])
                      for i, idx in enumerate(pred_idxs)]
        if label is None:
            return decode_out
        label = [(self.label_list[idx], 1.0) for idx in label]
        return decode_out, label


class TextClassifier(object):
    def __init__(self, args):
        self.cls_image_shape = [int(v) for v in args.cls_image_shape.split(",")]
        self.cls_batch_num = args.cls_batch_num
        self.cls_thresh = args.cls_thresh
        self.postprocess_op = ClsPostProcess(args.label_list)
        self.args = args

        if args.backend_type == "onnx":
            from utils.onnx_config import ONNXConfig
            self.predictor = ONNXConfig(args.cls_model_dir)
        elif args.backend_type == "rk_pc":
            from utils.rknn_config import RKNNConfigPC
            # config
            rknn_std = [[round(std * 255, 3) for std in [0.5, 0.5, 0.5]]]
            rknn_mean = [[round(mean * 255, 3) for mean in [0.5, 0.5, 0.5]]]
            self.predictor = RKNNConfigPC(model_path=args.cls_model_dir,
                                          mean_values=rknn_mean,
                                          std_values=rknn_std)
        elif args.backend_type == "rk_board":
            from utils.rknn_config import RKNNConfigBoard
            self.predictor = RKNNConfigBoard(args.cls_model_dir)

    def resize_norm_img(self, img, need=True):
        imgC, imgH, imgW = self.cls_image_shape
        h = img.shape[0]
        w = img.shape[1]
        ratio = w / float(h)
        if math.ceil(imgH * ratio) > imgW:
            resized_w = imgW
        else:
            resized_w = int(math.ceil(imgH * ratio))
        resized_image = cv2.resize(img, (resized_w, imgH))
        resized_image = resized_image.astype('float32')
        if need:
            if self.cls_image_shape[0] == 1:
                resized_image = resized_image / 255
                resized_image = resized_image[np.newaxis, :]
            else:
                resized_image = resized_image.transpose((2, 0, 1)) / 255
            resized_image -= 0.5
            resized_image /= 0.5
            padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
            padding_im[:, :, 0:resized_w] = resized_image
        else:
            padding_im = np.zeros((imgH, imgW, imgC), dtype=np.float32)
            padding_im[:, 0:resized_w, :] = resized_image
        return padding_im

    def __call__(self, img_list, is_onnx=False):
        img_list = copy.deepcopy(img_list)
        img_num = len(img_list)
        # Calculate the aspect ratio of all text bars
        width_list = []
        for img in img_list:
            width_list.append(img.shape[1] / float(img.shape[0]))
        # Sorting can speed up the cls process
        indices = np.argsort(np.array(width_list))

        cls_res = [['', 0.0]] * img_num
        batch_num = self.cls_batch_num

        for beg_img_no in range(0, img_num, batch_num):
            end_img_no = min(img_num, beg_img_no + batch_num)
            norm_img_batch = []
            max_wh_ratio = 0
            # for ino in range(beg_img_no, end_img_no):
            for ino in range(beg_img_no, end_img_no):
                h, w = img_list[indices[ino]].shape[0:2]
                wh_ratio = w * 1.0 / h
                max_wh_ratio = max(max_wh_ratio, wh_ratio)
            for ino in range(beg_img_no, end_img_no):
                norm_img = self.resize_norm_img(img_list[indices[ino]],need=self.args.backend_type == "onnx")
                norm_img = norm_img[np.newaxis, :]
                norm_img_batch.append(norm_img)

            norm_img_batch = np.concatenate(norm_img_batch)
            norm_img_batch = norm_img_batch.copy()

            outputs = self.predictor.infer(norm_img_batch)[0]
            outputs = np.argmax(outputs,axis=0)

            prob_out = outputs[0:end_img_no - beg_img_no, :]
            print(prob_out)
            cls_result = self.postprocess_op(prob_out)
            for rno in range(len(cls_result)):
                label, score = cls_result[rno]
                cls_res[indices[beg_img_no + rno]] = [label, score]
                if '180' in label and score > self.cls_thresh:
                    img_list[indices[beg_img_no + rno]] = cv2.rotate(
                        img_list[indices[beg_img_no + rno]], 1)
        print(cls_res)
        return img_list, cls_res
