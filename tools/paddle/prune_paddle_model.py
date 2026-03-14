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
import sys

import paddle
import paddle.base.core as core
import paddle.static as static


def prepend_feed_ops(program, feed_target_names):
    if len(feed_target_names) == 0:
        return

    global_block = program.global_block()
    feed_var = global_block.create_var(
        name="feed", type=core.VarDesc.VarType.FEED_MINIBATCH, persistable=True
    )

    for i, name in enumerate(feed_target_names):
        if not global_block.has_var(name):
            print(
                f"The input[{i}]: '{name}' doesn't exist in pruned inference program, which will be ignored in new saved model."
            )
            continue
        out = global_block.var(name)
        global_block._prepend_op(
            type="feed",
            inputs={"X": [feed_var]},
            outputs={"Out": [out]},
            attrs={"col": i},
        )


def append_fetch_ops(program, fetch_target_names):
    """
    In this palce, we will add the fetch op
    """
    global_block = program.global_block()
    fetch_var = global_block.create_var(
        name="fetch", type=core.VarDesc.VarType.FETCH_LIST, persistable=True
    )
    print(f"the len of fetch_target_names:{len(fetch_target_names)}")
    for i, name in enumerate(fetch_target_names):
        global_block.append_op(
            type="fetch",
            inputs={"X": [name]},
            outputs={"Out": [fetch_var]},
            attrs={"col": i},
        )


def insert_by_op_type(program, op_names, op_type):
    global_block = program.global_block()
    need_to_remove_op_index = []
    for i, op in enumerate(global_block.ops):
        if op.type == op_type:
            need_to_remove_op_index.append(i)
    for index in need_to_remove_op_index[::-1]:
        global_block._remove_op(index)
    program.desc.flush()

    if op_type == "feed":
        prepend_feed_ops(program, op_names)
    else:
        append_fetch_ops(program, op_names)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir", required=True, help="Path of directory saved the input model."
    )
    parser.add_argument(
        "--model_filename", required=True, help="The input model file name."
    )
    parser.add_argument(
        "--params_filename", required=True, help="The parameters file name."
    )
    parser.add_argument("--input_names", nargs="+", help="The inputs of pruned model.")
    parser.add_argument(
        "--output_names", required=True, nargs="+", help="The outputs of pruned model."
    )
    parser.add_argument(
        "--save_dir",
        required=True,
        help="Path of directory to save the new exported model.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    if len(set(args.output_names)) < len(args.output_names):
        print(
            "[ERROR] There's dumplicate name in --output_names, which is not allowed."
        )
        sys.exit(-1)

    paddle.enable_static()
    print("Start to load paddle model...")
    exe = static.Executor(paddle.CPUPlace())
    [program, feed_target_names, fetch_targets] = static.io.load_inference_model(
        args.model_dir,
        exe,
        model_filename=args.model_filename,
        params_filename=args.params_filename,
    )

    if args.input_names is not None:
        insert_by_op_type(program, args.input_names, "feed")
        feed_vars = [program.global_block().var(name) for name in args.input_names]
    else:
        feed_vars = [program.global_block().var(name) for name in feed_target_names]

    if args.output_names is not None:
        insert_by_op_type(program, args.output_names, "fetch")
        fetch_vars = [
            program.global_block().var(out_name) for out_name in args.output_names
        ]
    else:
        fetch_vars = list(fetch_targets)

    model_name = args.model_filename.split(".")[0]
    path_prefix = os.path.join(args.save_dir, model_name)
    static.io.save_inference_model(
        path_prefix=path_prefix,
        feed_vars=feed_vars,
        fetch_vars=fetch_vars,
        executor=exe,
        program=program,
    )
