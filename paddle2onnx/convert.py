# Copyright (c) 2022  PaddlePaddle Authors. All Rights Reserved.
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
import paddle
import paddle2onnx.paddle2onnx_cpp2py_export as c_p2o
from paddle2onnx.utils import logging, paddle_jit_save_configs

def export(model_file,
           params_file="",
           save_file=None,
           opset_version=9,
           auto_upgrade_opset=True,
           verbose=True,
           enable_onnx_checker=True,
           enable_experimental_op=True,
           enable_optimize=True,
           custom_op_info=None,
           deploy_backend="onnxruntime",
           calibration_file="",
           external_file="",
           export_fp16_model=False):
    deploy_backend = deploy_backend.lower()
    if custom_op_info is None:
        onnx_model_str = c_p2o.export(
            model_file, params_file, opset_version, auto_upgrade_opset, verbose,
            enable_onnx_checker, enable_experimental_op, enable_optimize, {},
            deploy_backend, calibration_file, external_file, export_fp16_model)
    else:
        onnx_model_str = c_p2o.export(
            model_file, params_file, opset_version, auto_upgrade_opset, verbose,
            enable_onnx_checker, enable_experimental_op, enable_optimize,
            custom_op_info, deploy_backend, calibration_file, external_file,
            export_fp16_model)
    if save_file is not None:
        with open(save_file, "wb") as f:
            f.write(onnx_model_str)
    else:
        return onnx_model_str


def dygraph2onnx(layer, save_file, input_spec=None, opset_version=9, **configs):
    # Get PaddleInference model file path
    dirname = os.path.split(save_file)[0]
    paddle_model_dir = os.path.join(dirname, "paddle_model_temp_dir")
    model_file = os.path.join(paddle_model_dir, "model.pdmodel")
    params_file = os.path.join(paddle_model_dir, "model.pdiparams")

    if os.path.exists(paddle_model_dir):
        if os.path.isfile(paddle_model_dir):
            logging.info("File {} exists, will remove it.".format(paddle_model_dir))
            os.remove(paddle_model_dir)
        if os.path.isfile(model_file):
            os.remove(model_file)
        if os.path.isfile(params_file):
            os.remove(params_file)
    save_configs = paddle_jit_save_configs(configs)
    paddle.jit.save(
        layer, os.path.join(paddle_model_dir, "model"), input_spec, **save_configs
    )
    logging.info("Static PaddlePaddle model saved in {}.".format(paddle_model_dir))
    if not os.path.isfile(params_file):
        params_file = ""

    if save_file is None:
        return export(model_file, params_file, save_file, opset_version)
    else:
        export(model_file, params_file, save_file, opset_version)
    logging.info("ONNX model saved in {}.".format(save_file))
