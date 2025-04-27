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
from paddle2onnx.utils import logging, paddle2onnx_export_configs
from contextlib import contextmanager
from paddle.decomposition import decomp
from paddle.base.executor import global_scope
import shutil
import traceback

PADDLE2ONNX_EXPORT_TEMP_DIR = None


def get_tmp_dir_and_file(model_filename, suffix=""):
    global PADDLE2ONNX_EXPORT_TEMP_DIR
    if PADDLE2ONNX_EXPORT_TEMP_DIR is None:
        PADDLE2ONNX_EXPORT_TEMP_DIR = tempfile.mkdtemp()
    model_file_path, _ = os.path.splitext(model_filename)
    new_model_file_path = os.path.join(
        PADDLE2ONNX_EXPORT_TEMP_DIR, os.path.basename(model_file_path) + suffix
    )
    new_model_file_name = new_model_file_path + ".json"
    new_params_file_name = new_model_file_path + ".pdiparams"
    return (
        model_file_path,
        new_model_file_path,
        new_model_file_name,
        new_params_file_name,
    )


def compare_programs(original_program, new_program):
    """Compares two pir programs' operations."""
    original_ops = [op.name() for op in original_program.global_block().ops]
    new_ops = [op.name() for op in new_program.global_block().ops]
    return original_ops == new_ops


def save_program(program, new_model_file_path):
    place = paddle.CPUPlace()
    exe = paddle.static.Executor(place)
    # Find feed and fetch operations
    feed, fetch = [], []
    # TODO(wangmingkai02): need double check it
    for op in program.global_block().ops:
        if op.name() == "pd_op.feed":
            feed.extend(op.results())
        if op.name() == "pd_op.fetch" or op.name() == "builtin.shadow_output":
            fetch.extend(op.operands_source())
    with paddle.pir_utils.IrGuard():
        paddle.static.save_inference_model(
            new_model_file_path, feed, fetch, exe, program=program
        )


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
    model_file_path, new_model_file_path, new_model_file_name, new_params_file_name = (
        get_tmp_dir_and_file(model_filename, "_decompose")
    )
    model = paddle.jit.load(model_file_path)
    new_program = model.program().clone()
    with decomp.prim_guard():
        decomp.decompose_dist_program(new_program)

    if compare_programs(model.program(), new_program):
        return model_filename

    load_parameter(new_program)
    save_program(new_program, new_model_file_path)
    return new_model_file_name


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
    optimize_tool="polygraphy",
):
    global PADDLE2ONNX_EXPORT_TEMP_DIR
    # check model_filename
    assert os.path.exists(
        model_filename
    ), f"Model file {model_filename} does not exist."
    if not os.path.exists(params_filename):
        logging.warning(
            f"Params file {params_filename} does not exist, "
            + "the exported onnx model will not contain weights."
        )
        params_filename = ""

    try:
        if model_filename.endswith(".pdmodel"):
            # translate old ir program to pir program
            logging.warning(
                "The .pdmodel file is deprecated in paddlepaddle 3.0"
                + " and will be removed in the future."
                + " Try to convert from .pdmodel file to json file."
            )
            (
                model_file_path,
                new_model_file_path,
                new_model_file_name,
                new_params_file_name,
            ) = get_tmp_dir_and_file(model_filename, "_pt")
            if os.path.exists(params_filename):
                place = paddle.CPUPlace()
                exe = paddle.static.Executor(place)
                with paddle.pir_utils.OldIrGuard():
                    [inference_program, feed_target_names, fetch_targets] = (
                        paddle.static.load_inference_model(model_file_path, exe)
                    )
                program = paddle.pir.translate_to_pir(inference_program.desc)
                # TODO(wangmingkai02): Do we need to call load_parameter(program) here?
                load_parameter(program)
                save_program(program, new_model_file_path)
                params_filename = new_params_file_name
                if not os.path.exists(new_params_file_name):
                    raise RuntimeError(
                        f"Program Tranlator failed due to params file {new_params_file_name} does not exist."
                    )
            else:
                with paddle.pir_utils.OldIrGuard():
                    program = paddle.load(model_filename)
                    pir_program = paddle.pir.translate_to_pir(program.desc)
                with paddle.pir_utils.IrGuard():
                    paddle.save(pir_program, new_model_file_name)
            if not os.path.exists(new_model_file_name):
                raise RuntimeError(
                    f"Program Tranlator failed due to json file {new_model_file_name} does not exist."
                )
            model_filename = new_model_file_name
            if verbose:
                logging.info("Complete the conversion from .pdmodel to json file.")

        if paddle.get_flags("FLAGS_enable_pir_api")["FLAGS_enable_pir_api"]:
            if dist_prim_all and auto_upgrade_opset:
                if verbose:
                    logging.info("Try to decompose program ...")
                # TODO(wangmingkai02): Do we need to update params_filename here?
                model_filename = decompose_program(model_filename)
                if verbose:
                    logging.info("Complete the decomposition of combined operators.")

        if verbose and PADDLE2ONNX_EXPORT_TEMP_DIR is not None:
            logging.info(
                f"Intermediate model and param files are saved at {PADDLE2ONNX_EXPORT_TEMP_DIR}"
            )

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
    except Exception as error:
        logging.error(f"Failed to convert PaddlePaddle model: {error}.")
        logging.error(traceback.print_exc())
    finally:
        if (
            os.environ.get("P2O_KEEP_TEMP_MODEL", "0").lower()
            not in [
                "1",
                "true",
                "on",
            ]
            and PADDLE2ONNX_EXPORT_TEMP_DIR is not None
        ):
            logging.warning(
                "Intermediate model and param files will be deleted,"
                + " if you want to keep them, please set env variable `P2O_KEEP_TEMP_MODEL` to True."
            )
            shutil.rmtree(PADDLE2ONNX_EXPORT_TEMP_DIR, ignore_errors=True)
            PADDLE2ONNX_EXPORT_TEMP_DIR = None

    if save_file is not None:
        # if optimize_tool == "onnxsim":
        #     try:
        #         logging.info(
        #             "Try to perform optimization on the ONNX model with Onnx Simplifier."
        #         )
        #         import io
        #         import onnx
        #         from onnxsim import simplify
        #         model_stream = io.BytesIO(onnx_model_str)
        #         onnx_model = onnx.load_model(model_stream)
        #         simplified_model, check = simplify(onnx_model)
        #         if check:
        #             onnx.save(simplified_model, save_file)
        #         else:
        #             logging.warning(f"Fail to simplify onnx model. Skip simplifying.")
        #     except Exception as error:
        #         logging.warning(
        #             f"Fail to simplify onnx model with error: {error}. Skip simplifying."
        #         )
        #         with open(save_file, "wb") as f:
        #             f.write(onnx_model_str)
        if optimize_tool == "onnxoptimizer":
            try:
                logging.info(
                    "Try to perform optimization on the ONNX model with onnxoptimizer."
                )
                import io
                import onnx
                import onnxoptimizer

                model_stream = io.BytesIO(onnx_model_str)
                onnx_model = onnx.load_model(model_stream)
                passes = [
                    "eliminate_deadend",
                    # "eliminate_identity", # some identity is useful in while block
                    "extract_constant_to_initializer",
                    "eliminate_unused_initializer",
                    "eliminate_duplicate_initializer",
                    "eliminate_nop_cast ",
                ]
                optimized_model = onnxoptimizer.optimize(onnx_model, passes)
                onnx.checker.check_model(optimized_model, full_check=True)
                onnx.save(optimized_model, save_file)
            except Exception as error:
                logging.warning(
                    f"Fail to optimize onnx model with error: {error}. Skip onnxoptimizer."
                )
                with open(save_file, "wb") as f:
                    f.write(onnx_model_str)
        elif optimize_tool == "polygraphy":
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
                onnx.checker.check_model(folded_model, full_check=True)
                origin_rank_list = []
                folded_rank_list = []
                for output in onnx_model.graph.output:
                    origin_rank_list.append(
                        len(output.type.tensor_type.shape.dim)
                        if output.type.tensor_type.HasField("shape")
                        else None
                    )
                for output in folded_model.graph.output:
                    folded_rank_list.append(
                        len(output.type.tensor_type.shape.dim)
                        if output.type.tensor_type.HasField("shape")
                        else None
                    )
                if len(origin_rank_list) != len(folded_rank_list) or any(
                    origin_rank_list[i] != folded_rank_list[i]
                    for i in range(len(origin_rank_list))
                ):
                    raise ValueError(
                        "The ranks of outputs in the original and folded model are inconsistent."
                    )
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
    paddle_model_dir = tempfile.mkdtemp()
    try:
        save_configs, export_configs = paddle2onnx_export_configs(configs)
        if paddle.get_flags("FLAGS_enable_pir_api")["FLAGS_enable_pir_api"]:
            model_file = os.path.join(paddle_model_dir, "model.json")
        else:
            model_file = os.path.join(paddle_model_dir, "model.pdmodel")
        paddle.jit.save(
            layer, os.path.join(paddle_model_dir, "model"), input_spec, **save_configs
        )
        if not os.path.isfile(model_file):
            raise ValueError("Failed to save static PaddlePaddle model.")
        logging.info("Static PaddlePaddle model saved in {}.".format(paddle_model_dir))
        params_file = os.path.join(paddle_model_dir, "model.pdiparams")
        if not os.path.isfile(params_file):
            params_file = ""
        export(model_file, params_file, save_file, opset_version, **export_configs)
    except Exception as err:
        logging.error(f"Failed to convert PaddlePaddle model due to {err}.")
    finally:
        if os.environ.get("P2O_KEEP_TEMP_MODEL", "0").lower() not in [
            "1",
            "true",
            "on",
        ]:
            logging.warning(
                "Static PaddlePaddle model will be deleted, if you want to keep it,"
                + " please set env variable `P2O_KEEP_TEMP_MODEL` to True."
            )
            shutil.rmtree(paddle_model_dir, ignore_errors=True)
