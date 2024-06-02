# Copyright (c) 2024  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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
import numpy as np
import onnxruntime

import paddle
from onnxbase import APIOnnx, randtool
import paddle2onnx

import unittest

class TestFP32ToFP16(unittest.TestCase):
    def test():
        pass

if __name__ == "__main__":
    # download resnet model
    if not os.path.exists("ResNet50_infer"):
        os.system("wget https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/inference/ResNet50_infer.tar && tar -xf ResNet50_infer.tar && rm -rf ResNet50_infer.tar")

    # generate fp16 model
    path = "ResNet50_infer/inference"
    # paddle.set_device("gpu")
    model = paddle.jit.load(path)
    model.float16()
    input_spec = [paddle.static.InputSpec(shape=[-1, 3, 224, 224], dtype='float16', name='inputs')]
    paddle.jit.save(model, 'ResNet50_infer/inference_fp16', input_spec=input_spec)

    # model.eval()
    # x = paddle.randn([1, 3, 224, 224], "float16")
    # output = model(x)
    # print(f"output: {output}")

    # convert to onnx
    os.system("paddle2onnx --model_dir ResNet50_infer --model_filename inference_fp16.pdmodel --params_filename inference_fp16.pdiparams --export_fp16_model True --save_file ResNet50_infer/resnet_fp16.onnx")
    # os.system("paddle2onnx --model_dir ResNet50_infer --model_filename inference.pdmodel --params_filename inference.pdiparams --export_fp16_model True --save_file ResNet50_infer/resnet_fp16.onnx")

    # valid precision
    # np.random.seed(10)
    # input_img = np.random.rand(1, 3, 224, 224).astype("float32")

    # onnx_file_name = "/wuzp/Paddle2ONNX/model/ResNet50_infer/resnet_fp32.onnx"
    # # providers = [("CUDAExecutionProvider")]
    # ort_session = onnxruntime.InferenceSession(onnx_file_name)

    # ort_inputs = {ort_session.get_inputs()[0].name: input_img}
    # ort_outputs = ort_session.run(None, ort_inputs)
    # print(f"ort_output: {ort_outputs}")
    # # print(onnxruntime.get_device())

    # onnx_file_name_fp16 = "/wuzp/Paddle2ONNX/model/resnet_fp16.onnx"
    # # providers = [("CUDAExecutionProvider")]
    # ort_session_fp16 = onnxruntime.InferenceSession(onnx_file_name_fp16)
    # input_img_fp16 = input_img.astype("float16")
    # ort_inputs_fp16 = {ort_session_fp16.get_inputs()[0].name: input_img_fp16}
    # ort_outputs_fp16 = ort_session_fp16.run(None, ort_inputs_fp16)
    # print(f"ort_outputs_fp16: {ort_outputs_fp16}")

    # # assert
    # np.testing.assert_allclose(
    #     ort_outputs_fp16[0], ort_outputs[0], rtol=1e-03, atol=1e-05
    # )
