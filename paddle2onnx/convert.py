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

from six import text_type as _text_type
import argparse
import sys


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        "-m",
        type=_text_type,
        default=None,
        help="Define model file path for tensorflow or onnx")
    parser.add_argument(
        "--save_dir",
        "-s",
        type=_text_type,
        default=None,
        help="Path to save translated model")
    parser.add_argument(
        "--version",
        "-v",
        action="store_true",
        default=False,
        help="Get version of paddle2onnx")
    parser.add_argument(
        "--onnx_opset",
        "-oo",
        type=int,
        default=10,
        help="Set onnx opset version to export")
    return parser


def convert(model_path, save_dir, opset_version=10):
    from paddle2onnx.decoder.paddle_decoder import PaddleDecoder
    from paddle2onnx.op_mapper.paddle_op_mapper import PaddleOpMapper
    import paddle.fluid as fluid
    model = PaddleDecoder(model_path, '__model__', '__params__')
    mapper = PaddleOpMapper()
    mapper.convert(
        model.program,
        save_dir,
        scope=fluid.global_scope(),
        opset_version=opset_version)


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
        print("paddle2onnx-{} with python>=3.5, paddlepaddle>=1.6.0\n".format(
            paddle2onnx.__version__))
        return

    assert args.save_dir is not None, "--save_dir is not defined"

    try:
        import paddle
        v0, v1, v2 = paddle.__version__.split('.')
        print("paddle.__version__ = {}".format(paddle.__version__))
        if v0 == '0' and v1 == '0' and v2 == '0':
            print("[WARNING] You are use develop version of paddlepaddle")
        elif int(v0) != 1 or int(v1) < 6:
            print("[ERROR] paddlepaddle>=1.6.0 is required")
            return
    except:
        print(
            "[ERROR] paddlepaddle not installed, use \"pip install paddlepaddle\""
        )

    assert args.model is not None, "--model should be defined while translating paddle model to onnx"
    convert(args.model, args.save_dir, opset_version=args.onnx_opset)


if __name__ == "__main__":
    main()
