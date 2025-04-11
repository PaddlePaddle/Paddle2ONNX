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
from paddle.decomposition import decomp
from paddle.base.executor import global_scope


def load_model(model_filename):
    """Loads the pir model from json file."""
    assert os.path.exists(
        model_filename
    ), f"Model file {model_filename} does not exist."
    if model_filename.endswith(".json"):
        model_filename = model_filename[:-5]
    return paddle.jit.load(model_filename)


def compare_programs(original_program, new_program):
    """Compares two pir programs' operations."""
    original_ops = [op.name() for op in original_program.global_block().ops]
    new_ops = [op.name() for op in new_program.global_block().ops]
    return original_ops == new_ops


def save_program(program, model_file):
    """Saves the decomposed program to a file."""
    place = paddle.CPUPlace()
    exe = paddle.static.Executor(place)

    tmp_dir = tempfile.mkdtemp()
    filename = os.path.basename(model_file) + "_decompose"
    filename_without_extension, _ = os.path.splitext(filename)
    save_dir = os.path.join(tmp_dir, filename_without_extension)

    # Find feed and fetch operations
    feed, fetch = [], []
    for op in program.global_block().ops:
        if op.name() == "pd_op.feed":
            feed.extend(op.results())
        if op.name() == "pd_op.fetch" or op.name() == "builtin.shadow_output":
            fetch.extend(op.operands_source())

    with paddle.pir_utils.IrGuard():
        paddle.static.save_inference_model(save_dir, feed, fetch, exe, program=program)

    new_model_file = save_dir + ".json"
    assert os.path.exists(
        new_model_file
    ), f"Pir Model file {new_model_file} does not exist."
    logging.info(f"Decomposed Model file path: {new_model_file}")
    return new_model_file


def load_parameter(program):
    params = []
    opts = []
    for var in program.list_vars():
        if var.is_parameter or var.get_defining_op().name() == "builtin.parameter":
            params.append(var)
        elif var.persistable and var.get_defining_op().name() == "pd_op.data":
            opts.append(var)
    vars_list = params + opts
    vars = [var for var in vars_list if var.persistable]

    if vars is None:
        return

    place = paddle.CPUPlace()
    exe = paddle.static.Executor(place)
    paddle.base.libpaddle.pir.create_loaded_parameter(
        vars, global_scope(), exe._default_executor
    )


def decompose_program(model_filename):
    """Decomposes the given pir program."""
    model = load_model(model_filename)
    new_program = model.program().clone()
    with decomp.prim_guard():
        decomp.decompose_dist_program(new_program)

    if compare_programs(model.program(), new_program):
        return model_filename

    load_parameter(new_program)
    return save_program(new_program, model_filename)


def get_old_ir_guard():
    # For old version of PaddlePaddle, do nothing guard is returned.
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
    dist_prim_all=False,
    verbose=False,
    enable_onnx_checker=True,
    enable_experimental_op=True,
    enable_optimize=True,
    custom_op_info=None,
    deploy_backend="onnxruntime",
    calibration_file="",
    external_file="",
    export_fp16_model=False,
    enable_polygraphy=True,
):
    # check model_filename
    assert os.path.exists(
        model_filename
    ), f"Model file {model_filename} does not exist."

    # translate old ir program to pir
    tmp_dir = tempfile.mkdtemp()
    dir_and_file, extension = os.path.splitext(model_filename)
    filename = os.path.basename(model_filename)
    filename_without_extension, _ = os.path.splitext(filename)
    save_dir = os.path.join(tmp_dir, filename_without_extension)
    if model_filename.endswith(".pdmodel"):
        if os.path.exists(model_filename) and os.path.exists(params_filename):
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
            with paddle.pir_utils.IrGuard():
                paddle.static.save_inference_model(
                    save_dir, feed, fetch, exe, program=program
                )
            model_filename = save_dir + ".json"
            params_filename = save_dir + ".pdiparams"
            assert os.path.exists(
                model_filename
            ), f"Pir Model file {model_filename} does not exist."
            assert os.path.exists(
                params_filename
            ), f"Pir Params file {params_filename} does not exist."
        else:
            with paddle.pir_utils.OldIrGuard():
                program = paddle.load(model_filename)
                pir_program = paddle.pir.translate_to_pir(program.desc)
            save_dir = os.path.join(tmp_dir, filename_without_extension)
            model_filename = save_dir + ".json"
            with paddle.pir_utils.IrGuard():
                paddle.save(pir_program, model_filename)
            assert os.path.exists(
                model_filename
            ), f"Pir Model file {model_filename} does not exist."
    if paddle.get_flags("FLAGS_enable_pir_api")["FLAGS_enable_pir_api"]:
        if dist_prim_all and auto_upgrade_opset:
            model_filename = decompose_program(model_filename)

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
        if enable_polygraphy:
            try:
                logging.info(
                    "Try to perform constant folding on the ONNX model with Polygraphy."
                )
                os.environ["POLYGRAPHY_AUTOINSTALL_DEPS"] = "1"
                import io
                import onnx
                from polygraphy.backend.onnx import fold_constants

                model_stream = io.BytesIO(onnx_model_str)
                onnx_model = onnx.load_model(model_stream)
                folded_model = fold_constants(onnx_model)
                onnx.save(folded_model, save_file)
            except Exception as error:
                logging.warning(
                    f"Fail to fold onnx model with error: {error}. Skip folding."
                )
                with open(save_file, "wb") as f:
                    f.write(onnx_model_str)
        else:
            with open(save_file, "wb") as f:
                f.write(onnx_model_str)
        logging.info("ONNX model saved in {}.".format(save_file))
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
