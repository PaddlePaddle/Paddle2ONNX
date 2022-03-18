#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import os
import traceback
import time
import sys
import argparse
from infer import ClassificationInfer


def str2list(v):
    if len(v) == 0:
        return None
    v = v.replace(" ", "")
    v = eval(v)
    return v


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_file', type=str, default="model.onnx")
    parser.add_argument(
        '--model_dir', type=str, default="inference", help='paddle_model_dir')
    parser.add_argument('--save_file', type=str, default="model.rknn")
    parser.add_argument('--image_path', type=str, help="image filename")
    parser.add_argument('--crop_size', default=224, help='crop_szie')
    parser.add_argument('--resize_size', default=256, help='resize_size')
    parser.add_argument('--diff_test', type=bool, default=False)
    parser.add_argument(
        '--backend_type',
        type=str,
        choices=["rk_hardware", "rk_pc", "onnxruntime", "paddle"],
        default="rk_pc")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    if args.backend_type != "paddle" and not os.path.exists(args.model_file):
        print(
            "[ERROR]：The provided model file: {} does not exist, please enter \"python deploy.py -h\" to view the help information.".
            format(args.model_file))
        return

    if not os.path.exists(args.image_path):
        print(
            "[ERROR]：The provided image file: {} does not exist, please enter \"python deploy.py -h\" to view the help information.".
            format(args.image_path))
        return

    if args.backend_type == "paddle":
        if not os.path.exists(args.model_dir):
            print(
                "[ERROR]：The provided model dir: {} does not exist, please enter \"python deploy.py -h\" to view the help information.".
                format(args.model_dir))
            return

    if args.diff_test:
        print(">>> Test Diff")
        infer_runner = ClassificationInfer()
        args.backend_type = "rk_pc"
        infer_runner.set_runner(args)
        rk_output = infer_runner.predict()
        infer_runner.release()

        args.backend_type = "onnxruntime"
        infer_runner.set_runner(args)
        infer_runner.predict()
        return

    if args.backend_type == "rk_hardware":
        print(">>> RK Hardware Infer")
        rk_hardware_runner = ClassificationInfer()
        rk_hardware_runner.set_runner(args)
        rk_hardware_output = rk_hardware_runner.predict()
        rk_hardware_runner.release()

    if args.backend_type == "rk_pc":
        print(">>> RK PC Infer")
        rk_pc_runner = ClassificationInfer()
        rk_pc_runner.set_runner(args)
        rk_pc_output = rk_pc_runner.predict()
        rk_pc_runner.release()

    if args.backend_type == "onnxruntime":
        print(">>> ONNXRuntime Infer")
        onnxruntime_runner = ClassificationInfer()
        onnxruntime_runner.set_runner(args)
        onnxruntime_runner.predict()

    if args.backend_type == "paddle":
        print(">>> Paddle Infer")
        paddle_runner = ClassificationInfer()
        paddle_runner.set_runner(args)
        paddle_runner.predict()


if __name__ == '__main__':
    main()
