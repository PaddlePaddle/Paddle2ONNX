#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
from __future__ import absolute_import

__version__ = "0.9.0"

import paddle
from .convert import dygraph2onnx, program2onnx
from .op_mapper import register_op_mapper
from typing import TypeVar

OP_WITHOUT_KERNEL_SET = {
    'feed', 'fetch', 'recurrent', 'go', 'rnn_memory_helper_grad',
    'conditional_block', 'while', 'send', 'recv', 'listen_and_serv',
    'fl_listen_and_serv', 'ncclInit', 'select', 'checkpoint_notify',
    'gen_bkcl_id', 'c_gen_bkcl_id', 'gen_nccl_id', 'c_gen_nccl_id',
    'c_comm_init', 'c_sync_calc_stream', 'c_sync_comm_stream',
    'queue_generator', 'dequeue', 'enqueue', 'heter_listen_and_serv',
    'c_wait_comm', 'c_wait_compute', 'c_gen_hccl_id', 'c_comm_init_hccl',
    'copy_cross_scope'
}


def run_convert(model, input_shape_dict=None, scope=None, opset_version=9):
    if isinstance(model, paddle.static.Program):
        if input_shape_dict is not None:
            for k, v in input_shape_dict.items():
                model.blocks[0].var(k).desc.set_shape(v)
            for i in range(len(model.blocks[0].ops)):
                if model.blocks[0].ops[i].type in OP_WITHOUT_KERNEL_SET:
                    continue
                model.blocks[0].ops[i].desc.infer_shape(model.blocks[0].desc)
        if scope is None:
            scope = paddle.static.global_scope()
        input_names = list()
        output_vars = list()
        for i in range(len(model.blocks[0].ops)):
            if model.blocks[0].ops[i].type == "feed":
                input_names.append(model.blocks[0].ops[i].output("Out")[0])
            if model.blocks[0].ops[i].type == "fetch":
                output_vars.append(model.blocks[0].var(model.blocks[0].ops[i]
                                                       .input("X")[0]))
        return program2onnx(
            model,
            scope,
            save_file=None,
            feed_var_names=input_names,
            target_vars=output_vars,
            opset_version=opset_version,
            enable_onnx_checker=True)
    elif isinstance(model, paddle.jit.TranslatedLayer):
        if input_shape_dict is not None:
            for k, v in input_shape_dict.items():
                model.program().blocks[0].var(k).desc.set_shape(v)
            for i in range(len(model.program().blocks[0].ops)):
                if model.program().blocks[0].ops[
                        i].type in OP_WITHOUT_KERNEL_SET:
                    continue
                model.program().blocks[0].ops[i].desc.infer_shape(model.program(
                ).blocks[0].desc)
        return dygraph2onnx(model, save_file=None, opset_version=opset_version)
    else:
        raise Exception(
            "Only support model loaded from paddle.static.load_inference_model() or paddle.jit.load()"
        )
