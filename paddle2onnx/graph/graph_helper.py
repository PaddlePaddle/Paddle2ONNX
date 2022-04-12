#   Copyright (c) 2020  PaddlePaddle Authors. All Rights Reserved.
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

import os
import paddle
import numpy as np
from paddle.fluid import core
from paddle.fluid.framework import Variable, program_guard
from paddle2onnx.utils import logging
from paddle2onnx.op_mapper import mapper_helper
from paddle2onnx.constant import dtypes
import itertools
import copy
import onnx


def prepend_feed_ops(inference_program,
                     feed_target_names,
                     feed_holder_name='feed'):
    if len(feed_target_names) == 0:
        return
    global_block = inference_program.global_block()
    feed_var = global_block.create_var(
        name=feed_holder_name,
        type=core.VarDesc.VarType.FEED_MINIBATCH,
        persistable=True)
    for i, name in enumerate(feed_target_names):
        if not global_block.has_var(name):
            raise ValueError(
                "The feed_var_names[{i}]: '{name}' doesn't exist in pruned inference program. "
                "Please check whether '{name}' is a valid feed_var name, or remove it from feed_var_names "
                "if '{name}' is not involved in the fetch_vars calculation.".
                format(
                    i=i, name=name))
        out = global_block.var(name)
        global_block._prepend_op(
            type='feed',
            inputs={'X': [feed_var]},
            outputs={'Out': [out]},
            attrs={'col': i})


def append_fetch_ops(inference_program,
                     fetch_target_names,
                     fetch_holder_name='fetch'):
    global_block = inference_program.global_block()
    fetch_var = global_block.create_var(
        name=fetch_holder_name,
        type=core.VarDesc.VarType.FETCH_LIST,
        persistable=True)
    for i, name in enumerate(fetch_target_names):
        global_block.append_op(
            type='fetch',
            inputs={'X': [name]},
            outputs={'Out': [fetch_var]},
            attrs={'col': i})


def get_program(program, feed_var_names, fetch_vars):
    global_block = program.global_block()
    need_to_remove_op_index = []
    for i, op in enumerate(global_block.ops):
        op.desc.set_is_target(False)
        if op.type == "feed" or op.type == "fetch":
            need_to_remove_op_index.append(i)
    for index in need_to_remove_op_index[::-1]:
        global_block._remove_op(index)
    program.desc.flush()
    program = program._prune_with_input(
        feeded_var_names=feed_var_names, targets=fetch_vars)
    program = program._inference_optimize(prune_read_op=True)
    fetch_var_names = [v.name for v in fetch_vars]
    prepend_feed_ops(program, feed_var_names)
    append_fetch_ops(program, fetch_var_names)
    return program


def static_quantize_pre_convert(graph):
    for name, node in graph.ctx.node_map.items():
        if not node.type.count("dequantize"):
            continue
        ipt = node.inputs["X"][0]
        for pre_name, pre_node in graph.ctx.node_map.items():
            pre_outputs = pre_node.outputs
            outputs = []
            for _, output in pre_outputs.items():
                outputs = outputs + output
            if ipt not in outputs:
                continue

            weight_input = pre_node.input('Filter', 0)
            if weight_input is None:
                weight_input = pre_node.input('Y', 0)
            update_param, weight = mapper_helper.get_param_from_paddle_graph(
                graph, weight_input)

            key = node.input('Scales', 0)
            if key is None:
                key = node.input('Scale', 0)
            _, weight_scale = mapper_helper.get_param_from_paddle_graph(graph,
                                                                        key)
            weight_scale = weight_scale / 127.0

            quant_axis = node.attr('quant_axis')
            if pre_node.type in ["conv2d", "depthwise_conv2d"]:
                if quant_axis is None:
                    quant_axis = 0
                new_weight = weight.transpose(1, 2, 3, 0) * weight_scale
                new_weight = new_weight.transpose(3, 0, 1, 2)
                update_param['data'] = new_weight
                graph.update_parameters(weight_input, update_param)
                input_shape = pre_node.input_shape('Filter', 0)
            else:
                if quant_axis is None:
                    quant_axis = 1
                new_weight = weight * weight_scale
                update_param['data'] = new_weight
                graph.update_parameters(weight_input, update_param)
                input_shape = pre_node.input_shape('Y', 0)

            scale_list = weight_scale.tolist()
            scale_node = graph.make_node(
                'Constant', dtype=dtypes.ONNX.FLOAT, value=scale_list)
            zero_list = [0] * len(weight_scale.tolist())
            zero_node = graph.make_node(
                'Constant', dtype=dtypes.ONNX.INT8, value=zero_list)
            graph.quantize_params_dict[weight_input] = [
                scale_node, zero_node, scale_list, zero_list, quant_axis
            ]

            quantize_node = graph.make_node(
                'QuantizeLinear',
                inputs=[weight_input, scale_node, zero_node],
                axis=quant_axis)
            if graph.is_graph_output(node.output('Out', 0)):
                graph.make_node(
                    'DequantizeLinear',
                    inputs=[quantize_node, scale_node, zero_node],
                    outputs=[node.output('Out', 0)],
                    axis=quant_axis)
            else:
                filter_node = graph.make_node(
                    'DequantizeLinear',
                    inputs=[quantize_node, scale_node, zero_node],
                    axis=quant_axis)
                try:
                    outputs = pre_node.output('Out', 0)
                except:
                    outputs = pre_node.output('Output', 0)
                graph.static_quantize_pre_convert_dict[outputs] = node.output(
                    'Out', 0)
                graph.static_quantize_pre_convert_dict[
                    weight_input] = filter_node


def static_quantize_post_process(graph):
    for name, node in graph.node_map.items():
        if node.type in ["QuantizeLinear", "DequantizeLinear"]:
            continue
        inputs = node.inputs
        outputs = node.outputs
        for index in range(len(inputs)):
            if inputs[index] in graph.static_quantize_pre_convert_dict:
                inputs[index] = graph.static_quantize_pre_convert_dict[inputs[
                    index]]
        for index in range(len(outputs)):
            if outputs[index] in graph.static_quantize_pre_convert_dict:
                outputs[index] = graph.static_quantize_pre_convert_dict[outputs[
                    index]]
