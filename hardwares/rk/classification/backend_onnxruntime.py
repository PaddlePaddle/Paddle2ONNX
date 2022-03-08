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
import onnxruntime as rt
sys.path.append("../..")
from common import BaseBackend


class ONNXRuntimeBackend(BaseBackend):
    def __init__(self):
        super(ONNXRuntimeBackend, self).__init__()

    def set_runner(self, config):
        self.config = config
        print('--> Load ONNX model')
        self.runner = rt.InferenceSession(self.config.model_file)
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
        data = np.transpose(data, (0, 3, 1, 2))
        self.inputs = dict()
        self.inputs[self.runner.get_inputs()[0].name] = data

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

        inner_postprocess(self.onnxruntime_infer[0], topk=5)

    def set_input(self):
        # resize the short edge to `resize_size`
        self.load_input = cv2.imread(self.config.image_path)
        self.preprocess()

    def predict(self):
        self.set_input()
        self.onnxruntime_infer = self.runner.run(None, self.inputs)
        self.postprocess()
        return self.outputs
