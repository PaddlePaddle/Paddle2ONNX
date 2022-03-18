#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import traceback
import time
import sys
import numpy as np
import cv2


class ClassificationInfer():
    def __init__(self):
        super(ClassificationInfer, self).__init__()

    def set_runner(self, config):
        self.config = config
        if self.config.backend_type == "onnxruntime":
            print('--> Load ONNX model')
            import onnxruntime as rt
            self.runner = rt.InferenceSession(self.config.model_file)
            print('done')

        if self.config.backend_type == "paddle":
            import paddle
            model_path = os.path.join(self.config.model_dir, "inference")
            self.runner = paddle.jit.load(model_path)

        if self.config.backend_type == "rk_hardware":
            from rknnlite.api import RKNNLite
            self.runner = RKNNLite()
            # load model
            ret = self.runner.load_rknn(self.config.model_file)
            if ret != 0:
                print('Load RKNN model failed')
            # init runtime environment
            print('--> Init runtime environment')
            ret = self.runner.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO)
            if ret != 0:
                print('Init runtime environment failed')

        if self.config.backend_type == "rk_pc":
            from rknn.api import RKNN
            self.runner = RKNN(verbose=True)
            print('--> config model')
            self.runner.config()
            print('done')
            # Load model
            print('--> Loading model')
            ret = self.runner.load_onnx(model=self.config.model_file)
            if ret != 0:
                print('Load model failed!')
                exit(ret)
            print('done')

            # Build model
            print('--> Building model')
            ret = self.runner.build(do_quantization=False)
            if ret != 0:
                print('Build model failed!')
                exit(ret)
            print('done')

            # Export rknn model
            print('--> Export rknn model')
            ret = self.runner.export_rknn(self.config.save_file)
            if ret != 0:
                print('Export rknn model failed!')
                exit(ret)
            print('done')

            # Init runtime environment
            print('--> Init runtime environment')
            ret = self.runner.init_runtime()
            if ret != 0:
                print('Init runtime environment failed!')
                exit(ret)
            print('done')

    def preprocess(self):
        """ Preprocess input image file
        Returns:
            preprocessed data(np.ndarray): Shape of [N, H, W, C]
        """

        def resize_by_short(im, resize_size):
            short_size = min(im.shape[0], im.shape[1])
            scale = resize_size / short_size
            new_w = int(round(im.shape[1] * scale))
            new_h = int(round(im.shape[0] * scale))
            return cv2.resize(
                im, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        def center_crop(im, crop_size):
            h, w, c = im.shape
            w_start = (w - crop_size) // 2
            h_start = (h - crop_size) // 2
            w_end = w_start + crop_size
            h_end = h_start + crop_size
            return im[h_start:h_end, w_start:w_end, :]

        def normalize(im, mean, std):
            im = im.astype("float32") / 255.0
            # to rgb
            im = im[:, :, ::-1]
            mean = np.array(mean).reshape((1, 1, 3)).astype("float32")
            std = np.array(std).reshape((1, 1, 3)).astype("float32")
            return (im - mean) / std

        resized_im = resize_by_short(self.load_input, self.config.resize_size)

        # crop from center
        croped_im = center_crop(resized_im, self.config.crop_size)

        # normalize
        normalized_im = normalize(croped_im, [0.485, 0.456, 0.406],
                                  [0.229, 0.224, 0.225])

        # transpose to NHWC
        data = np.expand_dims(normalized_im, axis=0)

        if self.config.backend_type == "onnxruntime":
            data = np.transpose(data, (0, 3, 1, 2))
            self.inputs = dict()
            self.inputs[self.runner.get_inputs()[0].name] = data
        if self.config.backend_type == "rk_pc":
            self.inputs = list()
            self.inputs = [data]
        if self.config.backend_type == "rk_hardware":
            self.inputs = list()
            self.inputs = [data]
        if self.config.backend_type == "paddle":
            data = np.transpose(data, (0, 3, 1, 2))
            self.inputs = data

    def postprocess(self):
        def inner_postprocess(result, topk=5):
            # choose topk index and score
            scores = result.flatten()
            topk_indices = np.argsort(-1 * scores)[:topk]
            topk_scores = scores[topk_indices]
            print("TopK Indices: ", topk_indices)
            print("TopK Scores: ", topk_scores)
            self.outputs = dict()
            self.outputs["Indices"] = topk_indices
            self.outputs["Scores"] = topk_scores

        inner_postprocess(self.infer_result[0], topk=5)

    def set_input(self):
        self.load_input = cv2.imread(self.config.image_path)
        self.preprocess()

    def predict(self):
        self.set_input()
        if self.config.backend_type == "rk_hardware":
            self.infer_result = self.runner.inference(inputs=self.inputs)
        if self.config.backend_type == "onnxruntime":
            self.infer_result = self.runner.run(None, self.inputs)
        if self.config.backend_type == "rk_pc":
            self.infer_result = self.runner.inference(inputs=self.inputs)
        if self.config.backend_type == "paddle":
            import paddle
            self.infer_result = [
                self.runner(paddle.to_tensor(self.inputs)).numpy()
            ]
        self.postprocess()
        return self.outputs

    def release(self):
        if self.config.backend_type in ["rk_pc", "rk_hardware"]:
            self.runner.release()
