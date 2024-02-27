# Copyright (c) 2020  PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import absolute_import
from six import text_type as _text_type
import argparse
import ast
import sys
import os
from paddle2onnx.utils import logging


def str2list(v):
    if len(v) == 0:
        return None
    v = v.replace(" ", "")
    v = eval(v)
    return v


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir",
        "-m",
        type=_text_type,
        default=None,
        help="PaddlePaddle model directory, if params stored in single file, you need define '--model_filename' and 'params_filename'."
    )
    parser.add_argument(
        "--model_filename",
        "-mf",
        type=_text_type,
        default=None,
        help="PaddlePaddle model's network file name, which under directory seted by --model_dir"
    )
    parser.add_argument(
        "--params_filename",
        "-pf",
        type=_text_type,
        default=None,
        help="PaddlePaddle model's param file name(param files combined in single file), which under directory seted by --model_dir."
    )
    parser.add_argument(
        "--save_file",
        "-s",
        type=_text_type,
        default=None,
        help="file path to save onnx model")
    parser.add_argument(
        "--opset_version",
        "-ov",
        type=int,
        default=9,
        help="set onnx opset version to export")
    parser.add_argument(
        "--deploy_backend",
        "-d",
        type=_text_type,
        default="onnxruntime",
        choices=["onnxruntime", "tensorrt", "rknn", "others"],
        help="Quantize model deploy backend, default onnxruntime.")
    parser.add_argument(
        "--save_calibration_file",
        type=_text_type,
        default="calibration.cache",
        help="The calibration cache for TensorRT deploy, default calibration.cache."
    )
    parser.add_argument(
        "--enable_onnx_checker",
        type=ast.literal_eval,
        default=True,
        help="whether check onnx model validity, default True")
    parser.add_argument(
        "--enable_paddle_fallback",
        type=ast.literal_eval,
        default=False,
        help="whether use PaddleFallback for custom op, default is False")
    parser.add_argument(
        "--version",
        "-v",
        action="store_true",
        default=False,
        help="get version of paddle2onnx")
    parser.add_argument(
        "--enable_auto_update_opset",
        type=ast.literal_eval,
        default=True,
        help="whether enable auto_update_opset, default is True")
    parser.add_argument(
        "--external_filename",
        type=_text_type,
        default=None,
        help="The filename of external_data when the model is bigger than 2G.")
    parser.add_argument(
        "--export_fp16_model",
        type=ast.literal_eval,
        default=False,
        help="Whether export FP16 model for ORT-GPU, default False")
    parser.add_argument(
        "--custom_ops",
        type=_text_type,
        default="{}",
        help="Ops that needs to be converted to custom op, e.g --custom_ops '{\"paddle_op\":\"onnx_op\"}', default {}"
    )
    return parser


def c_paddle_to_onnx(model_file,
                     params_file="",
                     save_file=None,
                     opset_version=7,
                     auto_upgrade_opset=True,
                     verbose=True,
                     enable_onnx_checker=True,
                     enable_experimental_op=True,
                     enable_optimize=True,
                     deploy_backend="onnxruntime",
                     calibration_file="",
                     external_file="",
                     export_fp16_model=False,
                     custom_ops={}):
    import paddle2onnx.paddle2onnx_cpp2py_export as c_p2o
    onnx_model_str = c_p2o.export(
        model_file, params_file, opset_version, auto_upgrade_opset, verbose,
        enable_onnx_checker, enable_experimental_op, enable_optimize,
        custom_ops, deploy_backend, calibration_file, external_file,
        export_fp16_model)
    if save_file is not None:
        with open(save_file, "wb") as f:
            f.write(onnx_model_str)
    else:
        return onnx_model_str


def main():
    if len(sys.argv) < 2:
        logging.info("Use \"paddle2onnx -h\" to print the help information")
        logging.info(
            "For more information, please follow our github repo below:")
        logging.info("Github: https://github.com/PaddlePaddle/paddle2onnx.git")
        return

    parser = arg_parser()
    args = parser.parse_args()

    if args.version:
        import paddle2onnx
        logging.info("paddle2onnx-{} with python>=3.6, paddlepaddle>=2.0.0".
                     format(paddle2onnx.__version__))
        return

    assert args.model_dir is not None, "--model_dir should be defined while translating paddle model to onnx"
    assert args.save_file is not None, "--save_file should be defined while translating paddle model to onnx"

    model_file = os.path.join(args.model_dir, args.model_filename)
    if args.params_filename is None:
        params_file = ""
    else:
        params_file = os.path.join(args.model_dir, args.params_filename)

    if args.external_filename is None:
        args.external_filename = "external_data"

    base_path = os.path.dirname(args.save_file)
    if base_path and not os.path.exists(base_path):
        os.mkdir(base_path)
    external_file = os.path.join(base_path, args.external_filename)

    custom_ops_dict = eval(args.custom_ops)

    calibration_file = args.save_calibration_file
    c_paddle_to_onnx(
        model_file=model_file,
        params_file=params_file,
        save_file=args.save_file,
        opset_version=args.opset_version,
        auto_upgrade_opset=args.enable_auto_update_opset,
        verbose=True,
        enable_onnx_checker=args.enable_onnx_checker,
        enable_experimental_op=True,
        enable_optimize=True,
        deploy_backend=args.deploy_backend,
        calibration_file=calibration_file,
        external_file=external_file,
        export_fp16_model=args.export_fp16_model,
        custom_ops=custom_ops_dict)
    logging.info("===============Make PaddlePaddle Better!================")
    logging.info("A little survey: https://iwenjuan.baidu.com/?code=r8hu2s")


if __name__ == "__main__":
    main()
