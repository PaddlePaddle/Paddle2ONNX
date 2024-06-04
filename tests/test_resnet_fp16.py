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
    model.eval()
    input_spec = [paddle.static.InputSpec(shape=[-1, 3, 224, 224], dtype='float16', name='inputs')]
    # paddle.jit.save(model, 'ResNet50_infer/inference_fp16', input_spec=input_spec)

    # convert to onnx
    paddle.onnx.export(model, "./resnet_fp16", input_spec=input_spec, export_fp16_model=True) # ONNX模型导出

    # valid precision
    np.random.seed(10)
    input_img = np.random.rand(1, 3, 224, 224).astype("float16")

    onnx_file_name = "./resnet_fp16.onnx"
    ort_session = onnxruntime.InferenceSession(onnx_file_name)

    ort_inputs = {ort_session.get_inputs()[0].name: input_img}
    ort_outputs = ort_session.run(None, ort_inputs)

    # resnet50 cannot be inferenced by half?
    model.float()
    paddle_input = paddle.to_tensor(input_img, dtype="float32")
    paddle_output = model(paddle_input)

    # assert
    np.testing.assert_allclose(
        paddle_output.numpy(), ort_outputs[0], rtol=1e-03, atol=1e-05
    )
