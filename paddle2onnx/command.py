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
import sys
import os
import paddle.fluid as fluid


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        "-m",
        type=_text_type,
        default=None,
        help="paddle model path, '__model__' and '__params__' files need under this path, which saved by 'paddle.fluid.io.save_inference_model'."
    )
    parser.add_argument(
        "--mode_type", "-mt", type=_text_type, default='program', help="")
    parser.add_argument(
        "--save_file",
        "-s",
        type=_text_type,
        default=None,
        help="file path to save onnx model")
    parser.add_argument(
        "--version",
        "-v",
        action="store_true",
        default=False,
        help="get version of paddle2onnx")
    parser.add_argument(
        "--opset_version",
        "-oo",
        type=int,
        default=9,
        help="set onnx opset version to export")
    return parser


def convert_inference_model_to_onnx(model_dir, save_dir, opset_version=10):
    # convert model save with 'paddle.fluid.io.save_inference_model'
    try:
        import paddle
        v0, v1, v2 = paddle.__version__.split('.')
        print("paddle.__version__ = {}".format(paddle.__version__))
        if v0 == '0' and v1 == '0' and v2 == '0':
            print("[WARNING] You are use develop version of paddlepaddle")
        elif int(v0) != 1 or int(v1) < 8:
            raise ImportError("paddlepaddle>=1.8.0 is required")
    except:
        print(
            "[ERROR] paddlepaddle not installed, use \"pip install paddlepaddle\""
        )

    from paddle2onnx import convert_program_to_onnx
    exe = fluid.Executor(fluid.CPUPlace())
    [program, feed, fetchs] = fluid.io.load_inference_model(
        model_dir,
        exe,
        model_filename='__model__',
        params_filename='__params__')
    convert_program_to_onnx(
        program, fluid.global_scope(), save_dir, opset_version=opset_version)


def main():
    if len(sys.argv) < 2:
        print("Use \"paddle2onnx -h\" to print the help information")
        print("For more information, please follow our github repo below:)")
        print("\nGithub: https://github.com/PaddlePaddle/paddle2onnx.git\n")
        return

    parser = arg_parser()
    args = parser.parse_args()

    if args.version:
        import paddle2onnx
        print("paddle2onnx-{} with python>=2.7, paddlepaddle>=1.8.0\n".format(
            paddle2onnx.__version__))
        return

    assert args.model is not None, "--model should be defined while translating paddle model to onnx"
    assert args.save_file is not None, "--save_file should be defined while translating paddle model to onnx"
    convert_inference_model_to_onnx(
        args.model, args.save_file, opset_version=args.opset_version)


if __name__ == "__main__":
    main()
