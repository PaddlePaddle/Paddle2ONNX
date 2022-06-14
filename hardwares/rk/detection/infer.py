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
import sys
import time
import traceback
from functools import reduce

import cv2
import numpy as np
from model_zoo.detection.utils.visualize import save_imgs
from picodet_postprocess import PicoDetPostProcess


class DetectionInfer():
    def __init__(self):
        super(DetectionInfer, self).__init__()

    def set_runner(self, config):
        self.config = config
        if self.config.backend_type == "onnxruntime":
            print('--> Load ONNX model')
            import onnxruntime as rt
            self.runner = rt.InferenceSession(self.config.model_file)
            print('done')

        if self.config.backend_type == "paddle":
            import paddle
            model_path = os.path.join(self.config.model_dir, "model")
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
            import onnx
            from onnx import helper
            from onnxsim import simplify
            self.runner = RKNN(verbose=True)
            print('--> config model')
            self.runner.config(target_platform='rk3568')
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
           
            preprocessed data(np.ndarray): Shape of [N, C, H, W]
        """

        def resize(im, resize_size):
            assert len(resize_size) == 2
            assert resize_size[0] > 0 and resize_size[1] > 0
            origin_shape = im.shape[:2]
            resize_h, resize_w = resize_size
            im_scale_y = resize_h / float(origin_shape[0])
            im_scale_x = resize_w / float(origin_shape[1])
            im = cv2.resize(
                im, None, None, fx=im_scale_x, fy=im_scale_y, interpolation=cv2.INTER_CUBIC)
            im_info = dict()
            im_info['scale_factor'] = np.array(
                [im_scale_y, im_scale_x]).astype('float32')
            im_info['im_shape'] = np.array(
                origin_shape).astype('float32')
            im_info['input_shape'] = np.array(
                resize_size).astype('float32')
            return im, im_info

        def normalize(im, mean, std):
            im = im.astype("float32") / 255.0
            # to rgb
            im = im[:, :, ::-1]
            mean = np.array(mean).reshape((1, 1, 3)).astype("float32")
            std = np.array(std).reshape((1, 1, 3)).astype("float32")
            return (im - mean) / std

        def pad_stride(im, stride):
            coarsest_stride = stride
            coarsest_stride = coarsest_stride
            if coarsest_stride <= 0:
                return im
            im_c, im_h, im_w = im.shape
            pad_h = int(np.ceil(float(im_h) / coarsest_stride) * coarsest_stride)
            pad_w = int(np.ceil(float(im_w) / coarsest_stride) * coarsest_stride)
            padding_im = np.zeros((im_c, pad_h, pad_w), dtype=np.float32)
            padding_im[:, :im_h, :im_w] = im
            return padding_im

        im = self.load_input
        self.im_info = dict()
        resized_im, self.im_info = resize(im, self.config.resize_size)
        normalized_im = normalize(resized_im, [0.485, 0.456, 0.406],
                                  [0.229, 0.224, 0.225])
        normalized_im = normalized_im.transpose((2, 0, 1))
        paded_im = pad_stride(normalized_im, 32)
        image = np.array((paded_im, )).astype('float32')


        if self.config.backend_type == "onnxruntime":
            self.inputs = dict()
            self.inputs[self.runner.get_inputs()[0].name] = image
        if self.config.backend_type == "rk_pc":
            image = np.transpose(image, (0, 2, 3, 1))
            self.inputs = [image]
        if self.config.backend_type == "rk_hardware":
            image = np.transpose(image, (0, 2, 3, 1))
            self.inputs = [image]
        if self.config.backend_type == "paddle":
            self.inputs = [image]

    def postprocess(self):
        # postprocess output of predictor
        np_score_list = self.infer_result[:4]
        np_boxes_list = self.infer_result[4:]
        postprocessor = PicoDetPostProcess(
            self.im_info['input_shape'],
            [self.im_info['im_shape']],
            [self.im_info['scale_factor']])
        np_boxes, np_boxes_num = postprocessor(np_score_list, np_boxes_list)
        self.outputs = dict(boxes=np_boxes, boxes_num=np_boxes_num)
        
    def set_input(self):
        self.load_input = cv2.imread(self.config.image_path)
        self.preprocess()

    def predict(self):
        self.set_input()
        if self.config.backend_type == "rk_hardware":
            self.infer_result = self.runner.inference(inputs=self.inputs)
            self.infer_result = [item.squeeze(-1) for item in self.infer_result]
        if self.config.backend_type == "onnxruntime":
            self.infer_result = self.runner.run(None, self.inputs)
        if self.config.backend_type == "rk_pc":
            self.infer_result = self.runner.inference(inputs=self.inputs)
        if self.config.backend_type == "paddle":
            import paddle
            paddle_inp = [paddle.to_tensor(a) for a in self.inputs]
            paddle_result = self.runner(*paddle_inp)
            self.infer_result = [t.numpy() for t in paddle_result]

        self.postprocess()
        save_imgs(
            self.outputs,
            self.config.image_path,
            output_dir='./output_dir',
            threshold=0.5,
            prefix=self.config.backend_type)
        return self.outputs

    def release(self):
        if self.config.backend_type in ["rk_pc", "rk_hardware"]:
            self.runner.release()

