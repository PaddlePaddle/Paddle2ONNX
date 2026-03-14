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
import paddle.base as base
import paddle.static as static


def process_old_ops_desc(program):
    for i in range(len(program.blocks[0].ops)):
        if program.blocks[0].ops[i].type == "matmul" and not program.blocks[0].ops[
            i
        ].has_attr("head_number"):
            program.blocks[0].ops[i]._set_attr("head_number", 1)


def infer_shape(program, input_shape_dict):
    paddle.enable_static()

    OP_WITHOUT_KERNEL_SET = {
        "feed",
        "fetch",
        "recurrent",
        "go",
        "rnn_memory_helper_grad",
        "conditional_block",
        "while",
        "send",
        "recv",
        "listen_and_serv",
        "fl_listen_and_serv",
        "ncclInit",
        "select",
        "checkpoint_notify",
        "gen_bkcl_id",
        "c_gen_bkcl_id",
        "gen_nccl_id",
        "c_gen_nccl_id",
        "c_comm_init",
        "c_sync_calc_stream",
        "c_sync_comm_stream",
        "queue_generator",
        "dequeue",
        "enqueue",
        "heter_listen_and_serv",
        "c_wait_comm",
        "c_wait_compute",
        "c_gen_hccl_id",
        "c_comm_init_hccl",
        "copy_cross_scope",
    }
    model_version = program.desc._version()
    paddle_version = paddle.__version__
    major_ver = model_version // 1000000
    minor_ver = (model_version - major_ver * 1000000) // 1000
    patch_ver = model_version - major_ver * 1000000 - minor_ver * 1000
    model_version = f"{major_ver}.{minor_ver}.{patch_ver}"
    if model_version != paddle_version:
        print(
            f"[WARNING] The model is saved by paddlepaddle v{model_version}, but now your paddlepaddle is version of {paddle_version}, this difference may cause error, it is recommend you reinstall a same version of paddlepaddle for this model"
        )
    for k, v in input_shape_dict.items():
        program.blocks[0].var(k).desc.set_shape(v)
    for i in range(len(program.blocks)):
        for j in range(len(program.blocks[0].ops)):
            if program.blocks[i].ops[j].type in OP_WITHOUT_KERNEL_SET:
                continue
            program.blocks[i].ops[j].desc.infer_shape(program.blocks[i].desc)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        required=True,
        help="Directory path to input model + model name without suffix.",
    )
    parser.add_argument(
        "--input_shape_dict", required=True, help="The new shape information."
    )
    parser.add_argument(
        "--save_path",
        required=True,
        help="Directory path to save model + model name without suffix.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    paddle.enable_static()
    input_shape_dict_str = args.input_shape_dict
    input_shape_dict = eval(input_shape_dict_str)
    print("Start to load paddle model...")
    exe = base.Executor(paddle.CPUPlace())
    [program, feed_target_names, fetch_targets] = static.io.load_inference_model(
        args.model_path, exe
    )
    process_old_ops_desc(program)
    infer_shape(program, input_shape_dict)

    feed_vars = [program.global_block().var(name) for name in feed_target_names]
    static.io.save_inference_model(
        args.save_path,
        feed_vars=feed_vars,
        fetch_vars=fetch_targets,
        executor=exe,
        program=program,
    )
