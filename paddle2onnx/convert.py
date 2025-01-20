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
import tempfile
import paddle2onnx.paddle2onnx_cpp2py_export as c_p2o
from paddle2onnx.utils import logging, paddle_jit_save_configs
from contextlib import contextmanager


def get_old_ir_guard():
    # For old version of PaddlePaddle, donothing guard is returned.
    @contextmanager
    def dummy_guard():
        yield

    if not hasattr(paddle, "pir_utils"):
        return dummy_guard
    pir_utils = paddle.pir_utils
    if not hasattr(pir_utils, "DygraphOldIrGuard"):
        return dummy_guard
    return pir_utils.DygraphOldIrGuard


def export(
    model_filename,
    params_filename,
    save_file=None,
    opset_version=7,
    auto_upgrade_opset=True,
    verbose=True,
    enable_onnx_checker=True,
    enable_experimental_op=True,
    enable_optimize=True,
    custom_op_info=None,
    deploy_backend="onnxruntime",
    calibration_file="",
    external_file="",
    export_fp16_model=False,
):
    # check model_filename
    assert os.path.exists(
        model_filename
    ), f"Model file {model_filename} does not exist."
    # check params_filename
    assert os.path.exists(
        params_filename
    ), f"Params file {params_filename} does not exist."

    # translate old ir program to pir
    if model_filename.endswith(".pdmodel"):
        dir_and_file, extension = os.path.splitext(model_filename)
        filename = os.path.basename(model_filename)
        filename_without_extension, _ = os.path.splitext(filename)
        tmp_dir = tempfile.mkdtemp()
        paddle.enable_static()
        place = paddle.CPUPlace()
        exe = paddle.static.Executor(place)
        with paddle.pir_utils.OldIrGuard():
            [inference_program, feed_target_names, fetch_targets] = (
                paddle.static.load_inference_model(dir_and_file, exe)
            )
        program = paddle.pir.translate_to_pir(inference_program.desc)
        for op in program.global_block().ops:
            if op.name() == "pd_op.feed":
                feed = op.results()
            if op.name() == "pd_op.fetch":
                fetch = op.operands_source()
        save_dir = os.path.join(tmp_dir, filename_without_extension)
        paddle.static.save_inference_model(save_dir, feed, fetch, exe, program=program)
        model_filename = save_dir + ".json"
        params_filename = save_dir + ".pdiparams"
        assert os.path.exists(
            model_filename
        ), f"Pir Model file {model_filename} does not exist."
        assert os.path.exists(
            params_filename
        ), f"Pir Params file {params_filename} does not exist."

    deploy_backend = deploy_backend.lower()
    if custom_op_info is None:
        onnx_model_str = c_p2o.export(
            model_filename,
            params_filename,
            opset_version,
            auto_upgrade_opset,
            verbose,
            enable_onnx_checker,
            enable_experimental_op,
            enable_optimize,
            {},
            deploy_backend,
            calibration_file,
            external_file,
            export_fp16_model,
        )
    else:
        onnx_model_str = c_p2o.export(
            model_filename,
            params_filename,
            opset_version,
            auto_upgrade_opset,
            verbose,
            enable_onnx_checker,
            enable_experimental_op,
            enable_optimize,
            custom_op_info,
            deploy_backend,
            calibration_file,
            external_file,
            export_fp16_model,
        )
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
    with get_old_ir_guard()():
        # In PaddlePaddle 3.0.0b2, PIR becomes the default IR, but PIR export still in development.
        # So we need to use the old IR to export the model, avoid make users confused.
        # In the future, we will remove this guard and recommend users to use PIR.
        paddle.jit.save(
            layer, os.path.join(paddle_model_dir, "model"), input_spec, **save_configs
        )
    logging.info("Static PaddlePaddle model saved in {}.".format(paddle_model_dir))
    if not os.path.isfile(params_file):
        params_file = ""

    export(model_file, params_file, save_file, opset_version)
    logging.info("ONNX model saved in {}.".format(save_file))
