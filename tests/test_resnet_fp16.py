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
import paddle2onnx
from paddle.inference import PrecisionType, PlaceType, convert_to_mixed_precision


def test_resnet_fp16_convert():
    # download resnet model
    if not os.path.exists("ResNet50_infer"):
        os.system(
            "wget https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/inference/ResNet50_infer.tar && tar -xf ResNet50_infer.tar && rm -rf ResNet50_infer.tar"
        )

    # generate fp16 model
    path = "ResNet50_infer"
    src_model = os.path.join(path, "inference.pdmodel")
    src_params = os.path.join(path, "inference.pdiparams")
    dst_model = os.path.join(path, "inference_fp16.pdmodel")
    dst_params = os.path.join(path, "inference_fp16.pdiparams")

    convert_to_mixed_precision(
        src_model,  # fp32 model path
        src_params,  # fp32 params path
        dst_model,  # mix precious model path
        dst_params,  # mix precious params path
        PrecisionType.Half,
        PlaceType.GPU,
        False,
    )

    # paddle.set_device("gpu")
    paddle.enable_static()
    path_fp16 = os.path.join(path, "inference_fp16")
    exe = paddle.static.Executor(paddle.CUDAPlace(0))
    [inference_program, feed_target_names, fetch_targets] = (
        paddle.static.load_inference_model(path_fp16, exe)
    )

    # infer paddle fp16
    np.random.seed(10)
    tensor_img = np.array(np.random.random((1, 3, 224, 224)), dtype=np.float16)
    results = exe.run(
        inference_program,
        feed={feed_target_names[0]: tensor_img},
        fetch_list=fetch_targets,
    )

    # convert to onnx
    input_spec = [
        paddle.static.InputSpec(shape=[-1, 3, 224, 224], dtype="float16", name="inputs")
    ]
    model_file = path_fp16 + ".pdmodel"
    params_file = path_fp16 + ".pdiparams"
    paddle2onnx.export(
        model_file, params_file, "./resnet_fp16.onnx", export_fp16_model=True
    )  # ONNX模型导出

    # valid precision
    onnx_file_name = "./resnet_fp16.onnx"
    ort_session = onnxruntime.InferenceSession(onnx_file_name)

    ort_inputs = {ort_session.get_inputs()[0].name: tensor_img}
    ort_outputs = ort_session.run(None, ort_inputs)

    # assert
    np.testing.assert_allclose(results[0], ort_outputs[0], rtol=2e-02, atol=2e-05)


if __name__ == "__main__":
    test_resnet_fp16_convert()
