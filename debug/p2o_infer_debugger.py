# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

import argparse
import paddle
import paddle2onnx
import numpy as np
from onnxruntime import InferenceSession
import os
import re
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
tests_dir = os.path.join(current_dir, "..", "tests")
sys.path.insert(0, tests_dir)
import onnxbase


def parse_arguments():
    def parse_shapes(s):
        pattern = r"\((.*?)\)"
        matches = re.findall(pattern, s)
        shapes = [list(map(int, match.split(","))) for match in matches]
        return shapes

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir", required=True, help="Path of directory saved the input model."
    )
    parser.add_argument(
        "--model_filename", required=True, help="The input model file name."
    )
    parser.add_argument(
        "--save_dir",
        required=True,
        help="Path of directory to save the new exported model.",
    )
    parser.add_argument(
        "--input_nums",
        required=False,
        default=0,
        type=int,
        help="Number of input data.",
    )
    parser.add_argument(
        "--input_shapes",
        type=parse_shapes,
        help='Comma-separated shapes for each input, e.g., "(3,4),(5,6,7)".',
    )
    parser.add_argument(
        "--input_dtypes",
        nargs="+",
        choices=["float64", "float32", "int64", "int32"],
        help="Data types of input tensors.",
    )
    args = parser.parse_args()
    if args.input_nums != 0 and not args.input_shapes:
        parser.error("--input_shapes is required when --input_nums is not 0.")
    if args.input_nums != 0 and len(args.input_shapes) != args.input_nums:
        parser.error(
            "--input_shapes must have the same number of elements as --input_nums."
        )
    if args.input_nums != 0 and not args.input_dtypes:
        parser.error("--input_dtypes is required when --input_nums is not 0.")
    if args.input_nums != 0 and len(args.input_dtypes) != args.input_nums:
        parser.error(
            "--input_dtypes must have the same number of elements as --input_nums."
        )

    return args


def gerenate_random_inputs(input_shapes: list, input_dtypes: list[str]):
    def str2dtype(dtype: str):
        map = {
            "float64": np.float64,
            "float32": np.float32,
            "int64": np.int64,
            "int32": np.int32,
        }
        return map[dtype]

    inputs = []
    np_dtype_list = list(map(str2dtype, input_dtypes))
    for idx, shape in enumerate(input_shapes):
        inputs.append(np.random.randn(*shape).astype(np_dtype_list[idx]))
    return tuple(inputs)


def compare_results(paddle_model_path: str, onnx_model_path: str, inputs_data: tuple):
    paddle_model = paddle.jit.load(paddle_model_path)
    expect = paddle_model(*inputs_data)
    session = InferenceSession(
        onnx_model_path,
        providers=["CPUExecutionProvider"],
    )
    input_names = session.get_inputs()
    input_feed = dict()
    for idx, input_name in enumerate(input_names):
        input_feed[input_name.name] = inputs_data[idx]
    result = session.run(output_names=None, input_feed=input_feed)
    onnxbase.compare(result[:1], expect, 1e-5, 1e-5)
    print("Successfully !!!!")


"""
def load_parameter(program : Program, exeutor : Executor):
    params, opts, = [], []
    for var in program.list_vars():
        if var.is_parameter or var.get_defining_op().name() == "builtin.parameter":
            params.append(var)
        elif var.persistable and var.get_defining_op().name() == "pd_op.data":
            opts.append(var)
    vars_list = params + opts
    vars = [var for var in vars_list if var.persistable]
    if vars is None:
        return
    paddle.base.libpaddle.pir.create_loaded_parameter(
        vars, global_scope(), exeutor._default_executor
    )

def save_program(program : Program, model_file : str):
    place = paddle.CPUPlace()
    exe = paddle.static.Executor(place)
    load_parameter(program, exe)

    tmp_dir = tempfile.mkdtemp()
    filename = os.path.basename(model_file) + "_debug"
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
    return new_model_file
"""


def main():
    args = parse_arguments()
    print("Inputs shapes: ", args.input_shapes)
    path = os.path.join(args.model_dir, args.model_filename)
    model = paddle.jit.load(path)
    program = model.program()
    assert program.num_blocks == 1, "Only support single block model."
    for idx, op in enumerate(program.blocks[0].ops):
        print(f"idx: {idx}, op: {op.name()}")
    left, right = 0, len(program.blocks[0].ops)
    skip_forward_op_list = ["pd_op.feed", "pd_op.data", "builtin.parameter"]
    skip_backward_op_list = ["pd_op.fetch"]
    white_list = ["pd_op.full", "pd_op.full_with_tensor", "pd_op.full_like"]
    offset = 0
    # for op in program.blocks[0].ops:
    #     print(op.name())

    while left < right:
        clone_program = program.clone()
        idx = (left + right) // 2 + offset
        if idx < left:
            break
        op = clone_program.blocks[0].ops[idx]
        if op.name() in skip_forward_op_list:
            left = idx + 1
        elif op.name() in skip_backward_op_list:
            right = idx - 1
        elif op.name() in white_list or op.name().startswith(
            "builtin."
        ):  # combine, split, slice
            offset = offset - 1
        else:
            try:
                offset = 0
                paddle.base.libpaddle.pir.append_shadow_outputs(
                    clone_program, op.results(), idx + 1, f"debug_output_{op.name()}_"
                )
                paddle2onnx.load_parameter(clone_program)
                new_model_file = paddle2onnx.save_program(clone_program, path)
                new_params_file = os.path.splitext(new_model_file)[0] + ".pdiparams"
                onnx_model_file = os.path.splitext(new_model_file)[0] + ".onnx"
                paddle2onnx.export(new_model_file, new_params_file, onnx_model_file)
                compare_results(
                    os.path.splitext(new_model_file)[0],
                    onnx_model_file,
                    gerenate_random_inputs(args.input_shapes, args.input_dtypes),
                )
            except (AssertionError, Exception) as err:
                print(f"Failed at index {idx}, op_name {op.name()}, error: {err}")
                right = idx - 1
            else:
                print(f"Success at index {idx}, op_name {op.name()}")
                left = idx + 1


if __name__ == "__main__":
    main()
