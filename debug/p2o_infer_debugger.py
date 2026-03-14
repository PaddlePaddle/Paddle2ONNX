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
import logging
import os
import queue
import re
import shutil
import sys
import tempfile
import traceback
from contextlib import contextmanager

import numpy as np
from onnxruntime import InferenceSession
from prune_onnx_model import prune_onnx_model

import paddle
import paddle2onnx

current_dir = os.path.dirname(os.path.abspath(__file__))
tests_dir = os.path.join(current_dir, "..", "tests")
sys.path.insert(0, tests_dir)
import onnxbase  # noqa: E402

SKIP_FORWARD_OP_LIST = ["pd_op.feed", "pd_op.data", "builtin.parameter"]
SKIP_BACKWARD_OP_LIST = ["pd_op.fetch", "builtin.shadow_output", "cf.yield"]
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
CANDIDATE_STATUS = None


def parse_arguments():
    def parse_shapes(s):
        pattern = r"\((.*?)\)"
        matches = re.findall(pattern, s)
        shapes = [list(map(int, match.split(","))) for match in matches]
        return shapes

    def parse_comma_separated_list(s):
        return [int(x.strip()) for x in s.split(",")]

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir", required=True, help="Path of directory saved the input model."
    )
    parser.add_argument(
        "--model_filename", required=True, help="The input model file name."
    )
    parser.add_argument(
        "--save_dir",
        # required=True,
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
        "--checked_op_ids",
        type=parse_comma_separated_list,
        help="Comma-separated ids of ops, e.g., 100,101,200.",
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
    parser.add_argument(
        "--traversal",
        default=False,
        action="store_true",
        help="Locate the issue by traversing from a specific op.",
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
    if args.traversal and not args.checked_op_ids:
        parser.error("--checked_op_ids is required when --traversal is set.")

    return args


def update_candidate_status(new_status: bool):
    global CANDIDATE_STATUS
    CANDIDATE_STATUS = new_status


def generate_random_inputs(input_shapes: list, input_dtypes: list[str]):
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
        if shape == [0]:
            shape = ()
        if input_dtypes[idx].startswith("int"):
            inputs.append(
                np.random.randint(1, 10, size=shape).astype(np_dtype_list[idx])
            )
        else:
            inputs.append(np.random.randn(*shape).astype(np_dtype_list[idx]))
    return tuple(inputs)


def save_and_export(program, model_file_path):
    temp_dir = tempfile.mkdtemp()
    new_model_file_path = os.path.join(
        temp_dir, os.path.basename(model_file_path) + "_debug"
    )
    new_model_file = new_model_file_path + ".json"
    new_params_file = new_model_file_path + ".pdiparams"
    paddle2onnx.load_parameter(program)
    paddle2onnx.save_program(program, new_model_file_path)
    origin_params_file = os.path.splitext(model_file_path)[0] + ".pdiparams"
    if os.path.exists(origin_params_file):
        shutil.copy(origin_params_file, new_params_file)
    if not os.path.exists(new_params_file):
        new_params_file = ""
    onnx_model_file = os.path.splitext(new_model_file)[0] + ".onnx"
    paddle2onnx.export(new_model_file, new_params_file, onnx_model_file, verbose=True)
    return new_model_file, onnx_model_file


def check_operator_with_print(
    program,
    model_file,
    input_shapes,
    input_dtypes,
    index_mapping,
    candidates,
    linear_search,
    output_num,
):
    skip_op_list = SKIP_FORWARD_OP_LIST + SKIP_BACKWARD_OP_LIST + WHITE_LIST
    temp_file_dir = ""

    @contextmanager
    def _redirect_paddle_output_to_file(
        paddle_model_file, log_file, inputs_data: tuple
    ):
        import pickle
        import subprocess
        import tempfile

        temp_filename = None
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            pickle.dump(inputs_data, temp_file)
            temp_filename = temp_file.name
            command = f"""
import pickle
import paddle
# 从临时文件中加载参数
with open('{temp_filename}', 'rb') as f:
    inputs_data = pickle.load(f)
    paddle_model = paddle.jit.load('{paddle_model_file}')
    paddle_model(*inputs_data)
"""
        try:
            with open(log_file, "w") as f:
                process = subprocess.Popen(
                    [sys.executable, "-c", command], stdout=f, stderr=subprocess.PIPE
                )
                _, stderr = process.communicate()
                if stderr:
                    print(stderr.decode(), file=sys.stderr)
                yield
        finally:
            if os.path.exists(temp_filename):
                os.remove(temp_filename)

    @contextmanager
    def _redirect_stdout_to_file(filename):
        original_stdout_fd = os.dup(sys.stdout.fileno())
        original_stdout = sys.stdout
        try:
            sys.stdout.flush()
            with open(filename, "w", encoding="utf-8") as f:
                os.dup2(f.fileno(), sys.stdout.fileno())
                sys.stdout = open(
                    os.dup(sys.stdout.fileno()), "w", encoding="utf-8", errors="ignore"
                )
                yield
        finally:
            sys.stdout.flush()
            os.dup2(original_stdout_fd, sys.stdout.fileno())
            sys.stdout.close()
            sys.stdout = original_stdout
            os.close(original_stdout_fd)

    def _compare_results(paddle_model_path, onnx_model_path, inputs_data: tuple):
        log_file = "./print.log"
        with _redirect_paddle_output_to_file(paddle_model_path, log_file, inputs_data):
            pass
        # paddle_model = paddle.jit.load(paddle_model_path)
        # with _redirect_stdout_to_file(log_file):
        #     paddle_model(*inputs_data)
        #     sys.stdout.flush()
        pattern = re.compile(
            r"Variable:.*?- shape:\s.*?\[(.*?)\].*?- dtype:\s*(\w+).*?- data:\s*\[(.*?)\].*?",
            flags=re.DOTALL,
        )
        # TODO(wangmingkai02): adjust n according to the number of print op
        n = 8
        shape_list, dtype, data_list = [], None, []
        with open(log_file, encoding="utf-8") as f:
            lines = []
            for _ in range(n):
                line = f.readline()
                if not line:
                    break
                lines.append(line.strip())
            text = "\n".join(lines)
            for match in pattern.finditer(text):
                shape, dtype, data = match.groups()
                shape_list = [int(x) for x in shape.split(",")] if shape != "" else []
                data_list = [float(x) for x in data.split(" ")] if data != "" else []

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
        input_feed = {}
        for idx, input_name in enumerate(input_names):
            input_feed[input_name.name] = inputs_data[idx]
        result = session.run(output_names=None, input_feed=input_feed)
        # construct expect data
        expect = paddle.to_tensor(data_list).astype(dtype)
        expect = paddle.reshape(expect, shape_list)
        onnxbase.compare(result[output_num:], expect, 1e-5, 1e-5)

    def _check_operator(program, block, idx):
        testing_op = block.ops[idx]
        op_results = []
        for i, res in enumerate(testing_op.results()):
            if not res.use_empty():
                op_results.append((i, res))
            else:
                logger.info(
                    "Skip the %d result of operator %s, op_id %d which is not used by other ops.",
                    i,
                    testing_op.name(),
                    testing_op.id(),
                )

        for i, op_result in op_results:
            logger.info(
                "Processing the %d result of op %s, op_id %d, using idx %d",
                i,
                testing_op.name(),
                testing_op.id(),
                idx,
            )
            paddle.base.libpaddle.pir.append_print(
                program,
                op_result,
                1,
                f"Print ({idx}, {testing_op.name()}) outputs:",
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
            nonlocal temp_file_dir
            temp_file_dir = os.path.dirname(new_model_file)
            # remove print op
            for _op in block.ops:
                if _op.name() == "pd_op.print":
                    block.remove_op(_op)
            # TODO(wangmingkai02): compare results
            _compare_results(
                os.path.splitext(new_model_file)[0],
                onnx_model_file,
                generate_random_inputs(input_shapes, input_dtypes),
            )

    def _binary_search(program, block):
        block_res = True
        left, right = 0, len(block.ops) - 1
        offset = 0
        while left <= right:
            idx = (left + right) // 2 + offset
            op = block.ops[idx]
            op_name = op.name()
            if idx < left or op_name in SKIP_FORWARD_OP_LIST:
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
                    _check_operator(program, block, idx)
                except Exception as err:
                    block_res = False
                    right = idx - 1
                    logger.debug(
                        "[Binary Search] Failed at idx %d, op_name %s, op_id %d, err %s\n%s",
                        idx,
                        op.name(),
                        op.id(),
                        str(err),
                        traceback.format_exc(),
                    )
                else:
                    left = idx - offset + 1
                    logger.debug(
                        "[Binary Search] Success at idx %d, op_name %s, op_id %d",
                        idx,
                        op.name(),
                        op.id(),
                    )
                finally:
                    offset = 0
                    nonlocal temp_file_dir
                    if os.path.exists(temp_file_dir):
                        shutil.rmtree(temp_file_dir)

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
                        "[Linear Search] Failed at idx %d, op_name %s, op_id %d, error: %s\n%s",
                        idx,
                        op.name(),
                        op.id(),
                        str(err),
                        traceback.format_exc(),
                    )
                else:
                    logger.debug(
                        "[Linear Search] Success at idx %d, op_name %s, op_id %d",
                        idx,
                        op.name(),
                        op.id(),
                    )
                finally:
                    nonlocal temp_file_dir
                    if os.path.exists(temp_file_dir):
                        shutil.rmtree(temp_file_dir)

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
                update_candidate_status(False)
                logger.debug(
                    "Failed at index %d, idx %d, op_name %s, op_id %d, error: %s\n%s",
                    index,
                    idx,
                    op.name(),
                    op.id(),
                    str(err),
                    traceback.format_exc(),
                )
            else:
                logger.debug(
                    "Success at index %d, idx %d, op_name %s, op_id %d",
                    index,
                    idx,
                    op.name(),
                    op.id(),
                )
            finally:
                if os.path.exists(temp_file_dir):
                    shutil.rmtree(temp_file_dir)

    else:
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


def check_operator_with_shadow_output(
    program,
    model_file,
    input_shapes,
    input_dtypes,
    index_mapping,
    candidates,
    linear_search,
    output_num,
):
    temp_file_dir = ""

    def _compare_results(
        paddle_model_path: str, onnx_model_path: str, inputs_data: tuple
    ):
        paddle_model = paddle.jit.load(paddle_model_path)
        expect = paddle_model(*inputs_data)
        session = InferenceSession(
            onnx_model_path,
            providers=["CPUExecutionProvider"],
        )
        input_names = session.get_inputs()
        input_feed = {}
        for idx, input_name in enumerate(input_names):
            input_feed[input_name.name] = inputs_data[idx]
        result = session.run(output_names=None, input_feed=input_feed)
        onnxbase.compare(result[:-output_num], expect, 1e-5, 1e-5)

    def _check_operator(program, model_file, idx, input_shapes, input_dtypes):
        op = program.blocks[0].ops[idx]
        op_results = []
        for i, res in enumerate(op.results()):
            if not res.use_empty():
                op_results.append(res)
            else:
                logger.info(
                    "Skip the %d result of operator %s, op_id %d which is not used by other ops.",
                    i,
                    op.name(),
                    op.id(),
                )
        paddle.base.libpaddle.pir.append_shadow_outputs(
            program, op_results, idx + 1, f"debug_output_{op.name()}_"
        )
        new_model_file, onnx_model_file = save_and_export(program, model_file)
        nonlocal temp_file_dir
        temp_file_dir = os.path.dirname(new_model_file)
        _compare_results(
            os.path.splitext(new_model_file)[0],
            onnx_model_file,
            generate_random_inputs(input_shapes, input_dtypes),
        )

    if candidates is not None and len(candidates) > 0:
        for index in candidates:
            if index_mapping[str(index)][1] != program.blocks[0]:
                logger.warning(
                    "Skip index %d which doesn't belong to global block", index
                )
                continue
            idx = index_mapping[str(index)][2]
            try:
                clone_program = program.clone()
                _check_operator(
                    clone_program, model_file, idx, input_shapes, input_dtypes
                )
            except (AssertionError, Exception) as err:
                update_candidate_status(False)
                logger.debug(
                    "Failed at index %d, idx %d, op_name %s, op_id  %d, error: %s\n%s",
                    index,
                    idx,
                    program.blocks[0].ops[idx].name(),
                    program.blocks[0].ops[idx].id(),
                    str(err),
                    traceback.format_exc(),
                )
            else:
                logger.debug(
                    "Success at index %d, idx %d, op_name %s, op_idx %d",
                    index,
                    idx,
                    program.blocks[0].ops[idx].name(),
                    program.blocks[0].ops[idx].id(),
                )
            finally:
                if os.path.exists(temp_file_dir):
                    shutil.rmtree(temp_file_dir)
    else:
        left, right = 0, len(program.blocks[0].ops) - 1
        offset = 0
        while left <= right:
            clone_program = program.clone()
            idx = (left + right) // 2 + offset
            op = clone_program.blocks[0].ops[idx]
            if idx < left or op.name() in SKIP_FORWARD_OP_LIST:
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
                    _check_operator(
                        clone_program, model_file, idx, input_shapes, input_dtypes
                    )
                except (AssertionError, Exception) as err:
                    right = idx - 1
                    logger.debug(
                        "Failed at idx %d, op_name %s, op_id  %d, error: %s\n%s",
                        idx,
                        op.name(),
                        op.id(),
                        str(err),
                        traceback.format_exc(),
                    )
                else:
                    left = idx - offset + 1
                    logger.debug(
                        "Success at idx %d, op_name %s, op_idx %d",
                        idx,
                        op.name(),
                        op.id(),
                    )
                finally:
                    offset = 0
                    if os.path.exists(temp_file_dir):
                        shutil.rmtree(temp_file_dir)


def locate_issue(
    program,
    index_mapping,
    model_file,
    input_shapes,
    input_dtypes,
    candidates: list[int] | None = None,
    has_cf=False,
    binary_search=False,
    output_num=1,
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
            output_num,
        )
    else:
        check_operator_with_shadow_output(
            program,
            model_file,
            input_shapes,
            input_dtypes,
            index_mapping,
            candidates,
            binary_search,
            output_num,
        )


def get_op_statistics(program):
    def _dfs(block, mapping):
        op_set = set()
        for idx, op in enumerate(block.ops):
            if op.name() == "pd_op.while":
                op_set |= _dfs(op.as_while_op().body(), mapping)
            elif op.name() == "pd_op.if":
                op_set |= _dfs(op.as_if_op().true_block(), mapping)
                op_set |= _dfs(op.as_if_op().false_block(), mapping)
            op_set.add(op.name())
            if str(str(op.id())) in mapping:
                raise ValueError(
                    f"Duplicate op found: {op.name()}, {mapping[str(op.id())]}"
                )
            mapping[str(op.id())] = (op, block, idx)
        return op_set

    def _get_mapping_and_uniq_set(program):
        # mapping: op_id -> (op, block, op_idx_in_block)
        index_mapping = {}
        ops = set()
        global_ops = set()
        global_res = []
        shadow_output_op_num = 0
        for block in program.blocks:
            ops |= _dfs(block, index_mapping)
        for idx, op in enumerate(program.blocks[0].ops):
            if op.name() == "builtin.shadow_output":
                shadow_output_op_num += 1
            if op.name() in global_ops:
                continue
            global_ops.add(op.name())
            global_res.append((idx, op.name()))
        return index_mapping, ops, global_res, shadow_output_op_num

    return _get_mapping_and_uniq_set(program)


def find_defined_op_index(
    index_list, index_mapping, max_depth=2, op_index_mapping=None
):
    new_index_mapping = {}
    if op_index_mapping is None:
        for k, v in index_mapping.items():
            new_index_mapping[v[0]] = int(k)
    else:
        new_index_mapping = op_index_mapping
    ret = []
    cur_res = []
    visited_op_set = set()
    q = queue.Queue()
    for index in index_list:
        _op = index_mapping[str(index)][0]
        q.put(_op)
    count = len(index_list)
    depth = 0
    count_next = 0
    while not q.empty():
        cur_op = q.get()
        if cur_op not in visited_op_set:
            cur_res.append(new_index_mapping[cur_op])
            visited_op_set.add(cur_op)
            for val in cur_op.operands():
                if val.source().get_defining_op() is None:
                    continue
                q.put(val.source().get_defining_op())
                count_next += 1
        count -= 1
        if count == 0:
            depth += 1
            if depth > max_depth:
                break
            count = count_next
            count_next = 0
            ret.append(cur_res)
            cur_res = []
    return ret


def find_used_op_index(index_list, index_mapping, op_index_mapping=None):
    """
    ret: dict[int, list[list[int]]]
    """
    ret = {}
    new_index_mapping = {}
    if op_index_mapping is None:
        for k, v in index_mapping.items():
            new_index_mapping[v[0]] = int(k)
    else:
        new_index_mapping = op_index_mapping
    for index in index_list:
        cur_op = index_mapping[str(index)][0]
        cur_ret = []
        for val in cur_op.results():
            cur_res = []
            # for uop in val.all_used_ops_in_same_block():
            for uop in val.all_used_ops():
                cur_res.append(new_index_mapping[uop])
            cur_ret.append(cur_res)
        ret[index] = cur_ret
    return ret


def locate_issue_by_traversal(
    index,
    index_mapping,
    program,
    model_file_path,
    input_shapes,
    input_dtypes,
    output_num,
):
    update_candidate_status(True)
    new_index_mapping = {}
    for k, v in index_mapping.items():
        new_index_mapping[v[0]] = int(k)
    op = index_mapping[str(index)][0]
    need_check_list = [index]
    if op.name() == "cf.yield":
        need_check_list = find_defined_op_index(
            index_list=[index],
            index_mapping=index_mapping,
            max_depth=2,
            op_index_mapping=new_index_mapping,
        )[1]
    q = queue.Queue()
    for op_id in need_check_list:
        q.put(op_id)

    error_op_id_list = []
    visited_op_id_set = set()
    while not q.empty():
        cur_op_id = q.get()
        if cur_op_id in visited_op_id_set:
            continue
        visited_op_id_set.add(cur_op_id)
        locate_issue(
            program=program,
            index_mapping=index_mapping,
            model_file=model_file_path,
            input_shapes=input_shapes,
            input_dtypes=input_dtypes,
            candidates=[cur_op_id],
            has_cf=index_mapping[str(cur_op_id)][1] != program.blocks[0],
            output_num=output_num,
        )
        if not CANDIDATE_STATUS:
            error_op_id_list.append(cur_op_id)
            next_check_list = find_defined_op_index(
                index_list=[cur_op_id],
                index_mapping=index_mapping,
                max_depth=2,
                op_index_mapping=new_index_mapping,
            )
            if len(next_check_list) > 1:
                next_check_list = next_check_list[1]
            for next_id in next_check_list:
                if index_mapping[str(next_id)][0].name() in [
                    "builtin.parameter",
                    "pd_op.data",
                ]:
                    continue
                q.put(next_id)
        update_candidate_status(True)

    # check op which uses the result of wrong op
    exclude_op_id_list = []
    used_op_error_id_list = []
    for op_id in error_op_id_list:
        used_list = find_used_op_index([op_id], index_mapping, new_index_mapping)[op_id]
        is_correct = True
        for each_value_used_list in used_list:
            for used_op_id in each_value_used_list:
                if (
                    used_op_id in error_op_id_list
                    or used_op_id in used_op_error_id_list
                ):
                    is_correct = False
                    break
                if used_op_id in visited_op_id_set:
                    continue
                visited_op_id_set.add(used_op_id)
                locate_issue(
                    program=program,
                    index_mapping=index_mapping,
                    model_file=model_file_path,
                    input_shapes=input_shapes,
                    input_dtypes=input_dtypes,
                    candidates=[used_op_id],
                    has_cf=index_mapping[str(used_op_id)][1] != program.blocks[0],
                )
                if not CANDIDATE_STATUS:
                    update_candidate_status(True)
                    used_op_error_id_list.append(used_op_id)
                    is_correct = False
                    break
            if not is_correct:
                break
        if is_correct:
            exclude_op_id_list.append(op_id)
    return error_op_id_list, exclude_op_id_list


def main():
    args = parse_arguments()
    logger.info("Start to locate issue...")
    model_file_path = os.path.join(args.model_dir, args.model_filename)
    model = paddle.jit.load(model_file_path)
    program = model.program()
    # TODO(wangmingkai02): Add a check for print op
    assert program.num_blocks == 1, "Only support single block model."
    logger.info("Initial Program: \n%s", str(program))
    if os.environ.get("FLAGS_print_ir", None) is not None and os.environ.get(
        "FLAGS_print_ir", None
    ).lower() in ["1", "true", "on"]:
        sys.exit(0)
    index_mapping, uniq_ops, global_uniq_ops, output_num = get_op_statistics(program)
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
    if args.traversal:
        if len(args.checked_op_ids) > 1:
            logger.warning(
                "Traverse only supports one checked op at most once!, ops %s will be ignored.",
                repr(args.checked_op_ids[1:]),
            )
        err_list, exclude_list = locate_issue_by_traversal(
            index=args.checked_op_ids[0],
            index_mapping=index_mapping,
            program=program,
            model_file_path=model_file_path,
            input_shapes=args.input_shapes,
            input_dtypes=args.input_dtypes,
            output_num=output_num,
        )
        logger.info("Error op id list:\n%s\n", ",".join([str(x) for x in err_list]))
        logger.info(
            "Exclude op id list:\n%s\n", ",".join([str(x) for x in exclude_list])
        )
    else:
        locate_issue(
            program,
            index_mapping,
            model_file_path,
            args.input_shapes,
            args.input_dtypes,
            args.checked_op_ids,
            args.has_control_flow,
            args.linear_search,
            output_num,
        )


if __name__ == "__main__":
    main()
