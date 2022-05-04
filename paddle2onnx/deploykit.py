# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
from __future__ import absolute_import
from paddle2onnx.utils import logging
try:
    import paddle2onnx.deploykit_cpp2py_export as deploykit_c
except:
    logging.warning(
        "[Paddle2ONNX][WARN] This package didn't compile with deploykit, refer https://github.com/PaddlePaddle/Paddle2ONNX/tree/deploykit/docs/zh/deploykit.md for more details."
    )

try:
    OrtBackendOption = deploykit_c.OrtBackendOption
except:
    logging.warning(
        "[Paddle2ONNX][WARN] This package didn't compile with onnxruntime backend, refer https://github.com/PaddlePaddle/Paddle2ONNX/tree/deploykit/docs/zh/deploykit.md for more details."
    )


class OrtBackend:
    """ Initialization for onnxruntime Backend

    Arguments:
        model_file: Path of model file, if the suffix of this file is ".onnx", will be loaded as onnx model; otherwise will be loaded as Paddle model.
        params_file: Path of parameters file, if loaded as onnx model or there's no parameter for Paddle model, set params_file to empty.
        verbose: Wheter to open Paddle2ONNX log while load Paddle model
    """

    def __init__(self, model_file, params_file="", option=None, verbose=False):
        try:
            self.backend = deploykit_c.OrtBackend()
        except:
            logging.error(
                "[ERROR] Cannot import OrtBackend from deploykit, please make sure you are using library is prebuilt with onnxruntime."
            )
        if option is None:
            option = deploykit_c.OrtBackendOption()
        if model_file.strip().endswith(".onnx"):
            self.backend.load_onnx(model_file.strip(), option)
        else:
            self.backend.load_paddle(model_file.strip(),
                                     params_file.strip(), option, verbose)

    def infer(self, inputs):
        input_names = list()
        input_arrays = list()
        for k, v in inputs.items():
            input_names.append(k)
            input_arrays.append(inputs[k])
        return self.backend.infer(input_names, input_arrays)


try:
    TrtBackendOption = deploykit_c.TrtBackendOption
except:
    logging.warning(
        "[Paddle2ONNX][WARN] This package didn't compile with TensorRT backend, refer https://github.com/PaddlePaddle/Paddle2ONNX/tree/deploykit/docs/zh/deploykit.md for more details."
    )


class TrtBackend:
    """ Initialization for tensorrt Backend

    Arguments:
        model_file: Path of model file, if the suffix of this file is ".onnx", will be loaded as onnx model; otherwise will be loaded as Paddle model.
        params_file: Path of parameters file, if loaded as onnx model or there's no parameter for Paddle model, set params_file to empty.
        verbose: Wheter to open Paddle2ONNX log while load Paddle model
    """

    def __init__(self, model_file, params_file="", option=None, verbose=False):
        try:
            self.backend = deploykit_c.TrtBackend()
        except:
            logging.error(
                "[ERROR] Cannot import TrtBackend from deploykit, please make sure you are using library is prebuilt with TensorRT."
            )
        if option is None:
            option = TrtBackendOption()
        if model_file.strip().endswith(".onnx"):
            self.backend.load_onnx(model_file.strip(), option)
        else:
            self.backend.load_paddle(model_file.strip(),
                                     params_file.strip(), option, verbose)

    def infer(self, inputs):
        input_names = list()
        input_arrays = list()
        for k, v in inputs.items():
            input_names.append(k)
            input_arrays.append(inputs[k])
        return self.backend.infer(input_names, input_arrays)
