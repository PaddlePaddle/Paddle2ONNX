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
import paddle.fluid as fluid
from paddle2onnx.utils import logging


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
        "--enable_onnx_checker",
        type=ast.literal_eval,
        default=False,
        help="whether check onnx model validity, if True, please 'pip install onnx'"
    )
    parser.add_argument(
        "--version",
        "-v",
        action="store_true",
        default=False,
        help="get version of paddle2onnx")
    return parser


def program2onnx(model_dir,
                 save_file,
                 model_filename=None,
                 params_filename=None,
                 opset_version=9,
                 enable_onnx_checker=False):
    try:
        import paddle
    except:
        logging.error(
            "paddlepaddle not installed, use \"pip install paddlepaddle\"")

    v0, v1, v2 = paddle.__version__.split('.')
    if v0 == '0' and v1 == '0' and v2 == '0':
        logging.warning("You are use develop version of paddlepaddle")
    elif int(v0) <= 1 and int(v1) < 8:
        raise ImportError("paddlepaddle>=1.8.0 is required")

    import paddle2onnx as p2o
    # convert model save with 'paddle.fluid.io.save_inference_model'
    if hasattr(paddle, 'enable_static'):
        paddle.enable_static()
    exe = fluid.Executor(fluid.CPUPlace())
    if model_filename is None and params_filename is None:
        [program, feed, fetchs] = fluid.io.load_inference_model(model_dir, exe)
    else:
        [program, feed, fetchs] = fluid.io.load_inference_model(
            model_dir,
            exe,
            model_filename=model_filename,
            params_filename=params_filename)
    p2o.program2onnx(
        program,
        fluid.global_scope(),
        save_file,
        opset_version=opset_version,
        enable_onnx_checker=enable_onnx_checker)


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
        logging.info("paddle2onnx-{} with python>=2.7, paddlepaddle>=1.8.0".
                     format(paddle2onnx.__version__))
        return

    assert args.model_dir is not None, "--model_dir should be defined while translating paddle model to onnx"
    assert args.save_file is not None, "--save_file should be defined while translating paddle model to onnx"
    program2onnx(
        args.model_dir,
        args.save_file,
        args.model_filename,
        args.params_filename,
        opset_version=args.opset_version,
        enable_onnx_checker=args.enable_onnx_checker)


if __name__ == "__main__":
    main()
