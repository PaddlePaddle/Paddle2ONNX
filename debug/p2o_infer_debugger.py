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
import os
import re
import sys
import logging
import shutil
import numpy as np
from onnxruntime import InferenceSession
import paddle
import paddle2onnx
from prune_onnx_model import prune_onnx_model
from contextlib import contextmanager

current_dir = os.path.dirname(os.path.abspath(__file__))
tests_dir = os.path.join(current_dir, "..", "tests")
sys.path.insert(0, tests_dir)
import onnxbase

SKIP_FORWARD_OP_LIST = ["pd_op.feed", "pd_op.data", "builtin.parameter"]
SKIP_BACKWARD_OP_LIST = ["pd_op.fetch", "pd_op.shadow_output", "cf.yield"]
WHITE_LIST = [
    "pd_op.full",
    "pd_op.full_with_tensor",
    "pd_op.full_like",
    "pd_op.full_int_array",
]

logger = logging.getLogger("p2o-logger")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    "[P2O INFER DEBUGGER %(asctime)s - %(levelname)s] %(message)s"
)
ch.setFormatter(formatter)
logger.addHandler(ch)
logger.propagate = False


def parse_arguments():
    def parse_shapes(s):
        pattern = r"\((.*?)\)"
        matches = re.findall(pattern, s)
        shapes = [list(map(int, match.split(","))) for match in matches]
        return shapes

    def parse_comma_separated_list(s):
        return [int(x) for x in s.split(",")]

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
    parser.add_argument(
        "--fixed_positions",
        type=parse_comma_separated_list,
        help="Comma-separated positions of ops, e.g., 100,101,200.",
    )
    parser.add_argument(
        "--has_control_flow",
        default=False,
        action="store_true",
        help="Whether the model has control flow op.",
    )
    parser.add_argument(
        "--linear_search",
        default=False,
        action="store_true",
        help="Whether check all ops by linear search.",
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
        dtype_map = {
            "float64": np.float64,
            "float32": np.float32,
            "int64": np.int64,
            "int32": np.int32,
        }
        return dtype_map[dtype]

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
    onnxbase.compare(result[:-1], expect, 1e-5, 1e-5)


def save_and_export(program, model_file):
    paddle2onnx.load_parameter(program)
    new_model_file = paddle2onnx.save_program(program, model_file)
    new_params_file = os.path.splitext(new_model_file)[0] + ".pdiparams"
    onnx_model_file = os.path.splitext(new_model_file)[0] + ".onnx"
    paddle2onnx.export(new_model_file, new_params_file, onnx_model_file)
    return new_model_file, onnx_model_file


def check_operator_with_shadow_output(
    program, model_file, idx, input_shapes, input_dtypes
):
    op = program.blocks[0].ops[idx]
    op_results = []
    for i, res in enumerate(op.results()):
        if not res.use_empty():
            op_results.append(res)
        else:
            logger.info(
                "Skip the %d result of operator %s which is not used by other ops.",
                i,
                op.name(),
            )
    paddle.base.libpaddle.pir.append_shadow_outputs(
        program, op_results, idx + 1, f"debug_output_{op.name()}_"
    )
    new_model_file, onnx_model_file = save_and_export(program, model_file)
    compare_results(
        os.path.splitext(new_model_file)[0],
        onnx_model_file,
        gerenate_random_inputs(input_shapes, input_dtypes),
    )
    shutil.rmtree(os.path.dirname(new_model_file))


def check_operator_with_print(
    program,
    model_file,
    input_shapes,
    input_dtypes,
    index_mapping,
    candidates,
    linear_search=False,
):
    skip_op_list = SKIP_FORWARD_OP_LIST + SKIP_BACKWARD_OP_LIST + WHITE_LIST

    @contextmanager
    def _redirect_stdout_to_file(filename):
        original_stdout_fd = os.dup(sys.stdout.fileno())
        try:
            with open(filename, "w") as f:
                os.dup2(f.fileno(), sys.stdout.fileno())
                sys.stdout.flush()
                yield
        finally:
            os.dup2(original_stdout_fd, sys.stdout.fileno())
            os.close(original_stdout_fd)

    def _compare_results(paddle_model_path, onnx_model_path, inputs_data: tuple):
        paddle_model = paddle.jit.load(paddle_model_path)
        with _redirect_stdout_to_file("./print.log"):
            paddle_model(*inputs_data)
        pattern = re.compile(
            r"Variable:.*?- shape:\s.*?\[(.*?)\].*?- dtype:\s*(\w+).*?- data:\s*\[(.*?)\].*?",
            flags=re.DOTALL,
        )
        # TODO(wangmingkai02): adjust n according to the number of print op
        n = 8
        shape_list, dtype, data_list = [], None, []
        with open("./print.log", "r", encoding="utf-8") as f:
            lines = []
            for _ in range(n):
                line = f.readline()
                if not line:
                    break
                lines.append(line.strip())
            text = "\n".join(lines)
            for match in pattern.finditer(text):
                shape, dtype, data = match.groups()
                shape_list = [int(x) for x in shape.split(",")]
                data_list = [float(x) for x in data.split(",")]

        # modify onnx model
        modified_onnx_model = prune_onnx_model(
            onnx_model_path,
            target_node_name="p2o.print",
            target_dims=shape_list,
            target_dtype=dtype,
        )
        session = InferenceSession(
            modified_onnx_model,
            providers=["CPUExecutionProvider"],
        )
        input_names = session.get_inputs()
        input_feed = dict()
        for idx, input_name in enumerate(input_names):
            input_feed[input_name.name] = inputs_data[idx]
        result = session.run(output_names=None, input_feed=input_feed)
        # print(result)
        # construct expect data
        expect = paddle.to_tensor(data_list).astype(dtype)
        expect = paddle.reshape(expect, shape_list)
        # TODO(wangmingkai02): adjust start pos of result
        onnxbase.compare(result[1:], expect, 1e-5, 1e-5)
        shutil.rmtree(os.path.dirname(onnx_model_path))
        return True

    def _check_operator(program, block, idx):
        op = block.ops[idx]
        op_results = []
        for i, res in enumerate(op.results()):
            if not res.use_empty():
                op_results.append(res)
            else:
                logger.info("Skip the %d result of operator %s.", i, op.name())

        paddle.base.libpaddle.pir.append_prints(
            program,
            op_results,
            1,
            f"Print ({idx}, {op.name()}) outputs:",
            -1,
            True,
            True,
            True,
            True,
            True,
            "FORWARD",
            True,
            idx + 1,
        )
        new_model_file, onnx_model_file = save_and_export(program, model_file)
        for op in block.ops:
            if op.name() == "pd_op.print":
                block.remove_op(op)
        # TODO(wangmingkai02): compare results
        return _compare_results(
            os.path.splitext(new_model_file)[0],
            onnx_model_file,
            gerenate_random_inputs(input_shapes, input_dtypes),
        )

    def _binary_search(program, block):
        block_res = True
        left, right = 0, len(block.ops) - 1
        offset = 0
        while left <= right:
            idx = (left + right) // 2 + offset
            op = block.ops[idx]
            op_name = op.name()
            if idx < left:
                left = idx - offset + 1
                offset = 0
            elif op_name in SKIP_FORWARD_OP_LIST:
                left = idx - offset + 1
                offset = 0
            elif op_name in SKIP_BACKWARD_OP_LIST:
                right = idx - 1
                offset = 0
            # combine, split, slice
            elif op_name in WHITE_LIST or op_name.startswith("builtin."):
                offset = offset - 1
            elif op_name == "pd_op.while":
                body_block = op.as_while_op().body()
                res = _binary_search(program, body_block)
                if res:
                    left = idx - offset + 1
                else:
                    right = idx - 1
                offset = 0
            elif op_name == "pd_op.if":
                true_block = op.as_if_op().true_block()
                res_0 = _binary_search(program, true_block)
                false_block = op.as_if_op().false_block()
                res_1 = _binary_search(program, false_block)
                if res_0 and res_1:
                    left = idx - offset + 1
                else:
                    right = idx - 1
                offset = 0
            else:
                try:
                    is_correct = _check_operator(program, block, idx)
                    if is_correct:
                        left = idx - offset + 1
                        logger.debug(
                            "[Binary Search] Success at index %d, op_name %s, op_id %d",
                            idx,
                            op.name(),
                            op.id(),
                        )
                    else:
                        block_res = False
                        right = idx - 1
                        logger.debug(
                            "[Binary Search] Failed at index %d, op_name %s, op_id %d",
                            idx,
                            op.name(),
                            op.id(),
                        )
                    offset = 0

                except Exception as err:
                    logger.error(
                        "[Binary Search] Errors occurred at index %d, op_name %s, op_id %d, error: %s",
                        idx,
                        op.name(),
                        op.id(),
                        str(err),
                    )
        return block_res

    def _linear_search(program, block):
        for idx, _ in enumerate(block.ops):
            op = block.ops[idx]
            op_name = op.name()
            if op_name == "pd_op.while":
                body_block = op.as_while_op().body()
                _linear_search(program, body_block)
            elif op_name == "pd_op.if":
                true_block = op.as_if_op().true_block()
                _linear_search(program, true_block)
                false_block = op.as_if_op().false_block()
                _linear_search(program, false_block)
            elif op_name in skip_op_list or op_name.startswith("builtin."):
                continue
            else:
                try:
                    _check_operator(program, block, idx)
                except Exception as err:
                    logger.debug(
                        "[Linear Search] Failed at index %d, op_name %s, op_id %d, error: %s",
                        idx,
                        op.name(),
                        op.id(),
                        str(err),
                    )
                else:
                    logger.debug(
                        "[Linear Search] Success at index %d, op_name %s, op_idx %d",
                        idx,
                        op.name(),
                        op.id(),
                    )

    def _check_block_ops(program, block):
        if linear_search:
            _linear_search(program, block)
        else:
            _binary_search(program, block)

    if candidates is not None and len(candidates) > 0:
        for index in candidates:
            if index_mapping[str(index)][1] == program.blocks[0]:
                logger.warning("Skip index %d which belongs to global block", index)
                continue
            op, block, idx = index_mapping[str(index)]
            try:
                _check_operator(program, block, idx)
            except Exception as err:
                logger.debug(
                    "Failed at index %d, idx %d, op_name %s, op_id %d, error: %s",
                    index,
                    idx,
                    op.name(),
                    op.id(),
                    str(err),
                )
            else:
                logger.debug(
                    "Success at index %d, idx %d, op_name %s, op_id %d",
                    index,
                    idx,
                    op.name(),
                    op.id(),
                )
        return

    # skip global block ops excluding while and if op
    for idx, _ in enumerate(program.blocks[0].ops):
        clone_program = program.clone()
        op = clone_program.blocks[0].ops[idx]
        if op.name() == "pd_op.while":
            body_block = op.as_while_op().body()
            _check_block_ops(clone_program, body_block)
        elif op.name() == "pd_op.if":
            true_block = op.as_if_op().true_block()
            _check_block_ops(clone_program, true_block)
            false_block = op.as_if_op().false_block()
            _check_block_ops(clone_program, false_block)
        else:
            continue


def locate_issue(
    program,
    model_file,
    input_shapes,
    input_dtypes,
    index_mapping,
    candidates: list[int] = None,
    has_cf=False,
    binary_search=False,
):
    if has_cf:
        check_operator_with_print(
            program,
            model_file,
            input_shapes,
            input_dtypes,
            index_mapping,
            candidates,
            binary_search,
        )
        return
    if candidates is not None and len(candidates) > 0:
        for index in candidates:
            if index_mapping[str(index)][1] != program.blocks[0]:
                logger.warning("Skip index %d which belongs to cf block", index)
                continue
            idx = index_mapping[str(index)][2]
            try:
                clone_program = program.clone()
                check_operator_with_shadow_output(
                    clone_program, model_file, idx, input_shapes, input_dtypes
                )
            except (AssertionError, Exception) as err:
                logger.debug(
                    "Failed at index %d, op_name %s, op_id  %d, error: %s",
                    idx,
                    program.blocks[0].ops[idx].name(),
                    program.blocks[0].ops[idx].id(),
                    str(err),
                )
            else:
                logger.debug(
                    "Success at index %d, op_name %s, op_idx %d",
                    idx,
                    program.blocks[0].ops[idx].name(),
                    program.blocks[0].ops[idx].id(),
                )
    else:
        left, right = 0, len(program.blocks[0].ops) - 1
        offset = 0
        while left <= right:
            clone_program = program.clone()
            idx = (left + right) // 2 + offset
            op = clone_program.blocks[0].ops[idx]
            if idx < left:
                left = idx - offset + 1
                offset = 0
            elif op.name() in SKIP_FORWARD_OP_LIST:
                left = idx - offset + 1
                offset = 0
            elif op.name() in SKIP_BACKWARD_OP_LIST:
                right = idx - 1
                offset = 0
            elif op.name() in WHITE_LIST or op.name().startswith(
                "builtin."
            ):  # combine, split, slice
                offset = offset - 1
            else:
                try:
                    check_operator_with_shadow_output(
                        clone_program, model_file, idx, input_shapes, input_dtypes
                    )
                except (AssertionError, Exception) as err:
                    right = idx - 1
                    logger.debug(
                        "Failed at index %d, op_name %s, op_id  %d, error: %s",
                        idx,
                        op.name(),
                        op.id(),
                        str(err),
                    )
                else:
                    left = idx - offset + 1
                    logger.debug(
                        "Success at index %d, op_name %s, op_idx %d",
                        idx,
                        op.name(),
                        op.id(),
                    )
                finally:
                    offset = 0


def get_op_statistics(program):
    def _dfs(block, count, mapping):
        op_set = set()
        for idx, op in enumerate(block.ops):
            if op.name() == "pd_op.while":
                count, op_set_tmp = _dfs(op.as_while_op().body(), count, mapping)
                op_set |= op_set_tmp
            elif op.name() == "pd_op.if":
                count, op_set_tmp = _dfs(op.as_if_op().true_block(), count, mapping)
                op_set |= op_set_tmp
                count, op_set_tmp = _dfs(op.as_if_op().false_block(), count, mapping)
                op_set |= op_set_tmp
            op_set.add(op.name())
            if str(count) in mapping:
                raise ValueError(
                    f"Duplicate op found: {op.name()}, {mapping[str(count)]}"
                )
            mapping[str(count)] = (op, block, idx)
            count += 1
        return count, op_set

    def _get_mapping_and_uniq_set(program):
        # map: index -> (block, op, op_idx_in_block)
        count = 0
        index_mapping = {}
        ops = set()
        global_ops = set()
        global_res = list()
        for block in program.blocks:
            count, ops_tmp = _dfs(block, count, index_mapping)
            ops |= ops_tmp
        for idx, op in enumerate(program.blocks[0].ops):
            if op.name() in global_ops:
                continue
            global_ops.add(op.name())
            global_res.append((idx, op.name()))
        return index_mapping, ops, global_res

    return _get_mapping_and_uniq_set(program)


def main():
    args = parse_arguments()
    logger.info("Start to locate issue...")
    model_file_path = os.path.join(args.model_dir, args.model_filename)
    model = paddle.jit.load(model_file_path)
    program = model.program()
    assert program.num_blocks == 1, "Only support single block model."
    index_mapping, uniq_ops, global_uniq_ops = get_op_statistics(program)
    logger.info(
        "*********************** uniq ops: %d *************************", len(uniq_ops)
    )
    for op_name in uniq_ops:
        logger.info("%s", op_name)
    logger.info(
        "*********************** uniq ops in global: %d *************************",
        len(global_uniq_ops),
    )
    for idx, op_name in global_uniq_ops:
        logger.info("%d, %s", idx, op_name)
    logger.info("*********************** index mapping:  *************************")
    for k, v in index_mapping.items():
        logger.info(
            "index: %s : (op: %s, idx: %d, global op: %d)",
            k,
            v[0].name(),
            v[2],
            v[1] == program.blocks[0],
        )

    locate_issue(
        program,
        model_file_path,
        args.input_shapes,
        args.input_dtypes,
        index_mapping,
        args.fixed_positions,
        args.has_control_flow,
        args.linear_search,
    )


if __name__ == "__main__":
    main()
