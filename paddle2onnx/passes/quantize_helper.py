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


def try_replacing_upstream_output(graph, upstream_output_name, output_name):
    if output_name in graph.quantize_params_dict.keys() and \
        len(graph.input_name_to_nodes()[upstream_output_name]) == 1 and \
        not graph.is_graph_output(upstream_output_name):
        graph.replace_output_of_all_nodes(upstream_output_name, output_name)
        if upstream_output_name in graph.tensor_to_be_quantize:
            graph.tensor_to_be_quantize.remove(upstream_output_name)
        return True
    return False


def remove_all_quantize_ops(graph):
    print("remove_all_quantize_ops")
    node_map = list(graph.node_map.items())
    for idx in range(len(node_map)):
        name, node = node_map[idx]
        if node.type not in ["QuantizeLinear"]:
            continue
        nodes_to_be_remove = [name]
        inputs = node.inputs[0]
        outputs = node.outputs[0]
        for dequant_idx in range(idx + 1, len(node_map)):
            inner_name, inner_node = node_map[dequant_idx]
            inner_inputs = inner_node.inputs
            inner_outputs = inner_node.outputs
            if outputs in inner_inputs:
                nodes_to_be_remove.append(inner_name)
                outputs = inner_outputs[0]
                break

        for op_idx in range(dequant_idx + 1, len(node_map)):
            op_name, op_node = node_map[op_idx]
            op_inputs = op_node.inputs
            if outputs in op_inputs:
                for input_idx in range(len(op_inputs)):
                    if op_node.inputs[input_idx] == outputs:
                        op_node.inputs[input_idx] = inputs
                        op_node.set_inputs(op_node.inputs)
                        graph.update_node(op_node)

        graph.remove_node_by_name(nodes_to_be_remove[0])
        graph.remove_node_by_name(nodes_to_be_remove[1])

    return graph


def merge_conv_add(graph):
    node_map = list(graph.node_map.items())
    for idx in range(len(node_map)):
        conv_node_name, conv_node = node_map[idx]
        if conv_node.type not in ["Conv"]:
            continue
        conv_outputs = conv_node.outputs[0]
        for bn_idx in range(idx + 1, len(node_map)):
            add_node_name, add_node = node_map[bn_idx]
            if add_node.type in ["Conv"]:
                break
            if add_node.type not in ["Add"]:
                continue
            add_inputs = add_node.inputs
            add_outputs = add_node.outputs[0]
            if conv_outputs not in add_inputs:
                continue
            input_name_to_nodes = graph.input_name_to_nodes()
            if add_outputs in input_name_to_nodes and len(input_name_to_nodes[
                    add_outputs]) > 1:
                continue

            bias_node = add_node.inputs[0] if conv_node.outputs[
                0] == add_node.inputs[1] else add_node.inputs[1]
            output_name_from_nodes = graph.output_name_from_nodes()
            if bias_node not in output_name_from_nodes:
                continue
            add_input_node = output_name_from_nodes[bias_node][0]
            bias_weight = None
            bias_node = None
            if add_input_node.type in ["Reshape"]:
                bias_node = add_input_node.inputs[0]
                _, bias_weight = mapper_helper.get_param_from_paddle_graph(
                    graph, bias_node)
                if bias_weight is None:
                    break
            graph.remove_node(add_input_node)

            topk_data_sort = np.amax(np.abs(bias_weight))
            scale = topk_data_sort / 127.0
            scale_list = scale.tolist()
            if not isinstance(scale_list, list):
                scale_list = [scale_list]
            scale_node = graph.make_node(
                'Constant', dtype=dtypes.ONNX.FLOAT, value=scale_list[0])
            zero_node = graph.make_node(
                'Constant', dtype=dtypes.ONNX.INT8, value=0)
            if bias_node not in graph.quantize_params_dict:
                graph.quantize_params_dict[
                    bias_node] = [scale_node, zero_node, scale_list, [0], 0]
            conv_node.inputs.append(bias_node)
            conv_node.outputs = add_node.outputs
            graph.update_node(conv_node)
            graph.remove_node_by_name(add_node_name)
    return graph


def merge_conv_bn(graph):
    print("merge_conv_bn")
    node_map = list(graph.node_map.items())
    for idx in range(len(node_map)):
        conv_node_name, conv_node = node_map[idx]
        if conv_node.type not in ["Conv"]:
            continue

        conv_outputs = conv_node.outputs[0]
        for bn_idx in range(idx + 1, len(node_map)):
            bn_node_name, bn_node = node_map[bn_idx]
            if bn_node.type in ["Conv"]:
                break
            if bn_node.type not in ["BatchNormalization"]:
                continue
            bn_inputs = bn_node.inputs
            bn_outputs = bn_node.outputs
            if conv_outputs not in bn_inputs:
                continue
            input_name_to_nodes = graph.input_name_to_nodes()
            if not conv_outputs in input_name_to_nodes.keys():
                continue
            if len(input_name_to_nodes[conv_outputs]) > 1:
                continue
            conv_weight_node = conv_node.inputs[1]
            weight_params, conv_weight = mapper_helper.get_param_from_paddle_graph(
                graph, conv_weight_node)

            bn_scale_node = bn_node.inputs[1]
            _, bn_scale = mapper_helper.get_param_from_paddle_graph(
                graph, bn_scale_node)
            bn_bias_node = bn_node.inputs[2]
            _, bn_bias = mapper_helper.get_param_from_paddle_graph(graph,
                                                                   bn_bias_node)
            bn_mean_node = bn_node.inputs[3]
            _, bn_mean = mapper_helper.get_param_from_paddle_graph(graph,
                                                                   bn_mean_node)
            bn_var_node = bn_node.inputs[4]
            _, bn_var = mapper_helper.get_param_from_paddle_graph(graph,
                                                                  bn_var_node)
            epsilon = bn_node.attr("epsilon")
            momentum = bn_node.attr("momentum")

            conv_bias = np.zeros((bn_bias.shape[0]))
            conv_bias_node = conv_weight_node + ".merged.bias"
            if len(conv_node.inputs) == 3:
                conv_bias_node = conv_node.inputs[2]
                _, conv_bias = mapper_helper.get_param_from_paddle_graph(
                    graph, conv_bias_node)

            alpha = bn_scale / np.sqrt(bn_var + epsilon)

            new_bias = conv_bias * alpha + (bn_scale - bn_mean * alpha)
            new_weight = conv_weight.transpose(1, 2, 3, 0) * alpha
            new_weight = new_weight.transpose(3, 0, 1, 2)
            weight_params["data"] = new_weight
            graph.update_parameters(conv_weight_node, weight_params)

            topk_data_sort = np.amax(np.abs(new_bias))
            scale = topk_data_sort / 127.0
            scale_list = scale.tolist()
            if not isinstance(scale_list, list):
                scale_list = [scale_list]
            scale_node = graph.make_node(
                'Constant', dtype=dtypes.ONNX.FLOAT, value=scale_list[0])
            zero_node = graph.make_node(
                'Constant', dtype=dtypes.ONNX.INT32, value=0)

            new_bias = new_bias / scale

            bias_params = copy.copy(weight_params)
            bias_params['shape'] = [new_bias.shape[0]]
            bias_params["data"] = new_bias.astype("int32")
            bias_params['dtype'] = paddle.int32
            graph.update_parameters(conv_bias_node, bias_params)

            conv_node.inputs.append(conv_bias_node)
            conv_node.outputs = bn_node.outputs
            graph.update_node(conv_node)
            graph.remove_node_by_name(bn_node_name)
            if graph.quantize_model_mode in ["float"]:
                return graph
            graph.quantize_params_dict[
                conv_bias_node] = [scale_node, zero_node, scale_list, [0], 0]
    return graph


def add_q_dq(graph):
    print("add_q_dq")
    node_map = list(graph.node_map.items())
    for idx in range(len(node_map)):
        name, node = node_map[idx]
        if node.type in ["Clip", "Relu"]:
            if try_replacing_upstream_output(graph, node.inputs[0],
                                             node.outputs[0]):
                graph.remove_node_by_name(name)
            else:
                graph.tensor_to_be_quantize.append(node.inputs[0])
                graph.tensor_to_be_quantize.append(node.outputs[0])
        if len(graph.quantize_params_dict) == 0:
            continue
        if node.type in [
                "Reshape", "Transpose", "Squeeze", "Unsqueeze", "AveragePool"
        ]:
            graph.tensor_to_be_quantize.append(node.inputs[0])
            graph.tensor_to_be_quantize.append(node.outputs[0])

        if node.type in ["Conv"]:
            graph.tensor_to_be_quantize.append(node.inputs[0])
            graph.tensor_to_be_quantize.append(node.inputs[1])
            graph.tensor_to_be_quantize.append(node.outputs[0])
            if len(node.inputs) == 3:
                graph.tensor_to_be_quantize.append(node.inputs[2])

        if node.type in ["Resize", "MaxPool"]:
            graph.tensor_to_be_quantize.append(node.inputs[0])
            graph.tensor_to_be_quantize.append(node.outputs[0])

        if node.type in ["Concat", "MatMul"]:
            tensors_to_quantize = itertools.chain(node.inputs, node.outputs)
            for tensor_name in tensors_to_quantize:
                graph.tensor_to_be_quantize.append(tensor_name)

        if node.type in [
                "ArgMax", "Mul", "LeakyRelu", "Sigmoid", "GlobalAveragePool",
                "Split", "Pad", "Add"
        ]:
            tensors_to_quantize = itertools.chain(node.inputs, node.outputs)
            for tensor_name in tensors_to_quantize:
                graph.tensor_to_be_quantize.append(tensor_name)

    graph.tensor_to_be_quantize = list(set(graph.tensor_to_be_quantize))
    for tensor in graph.tensor_to_be_quantize:
        if tensor not in graph.quantize_params_dict:
            print(">>>[Quantize] Find unquantize tensor: ", tensor)
            continue
        quantize_nodes = graph.quantize_params_dict[tensor]
        assert len(
            quantize_nodes
        ) == 5, "The len of quantize_nodes must be 5, but get length is: {}".format(
            len(quantize_nodes))
        scale_node = quantize_nodes[0]
        zero_node = quantize_nodes[1]
        quant_axis = quantize_nodes[4]
        if quant_axis is None:
            quant_axis = -1

        if tensor.count(".merged.bias"):
            dequantize_node = graph.make_node(
                'DequantizeLinear',
                inputs=[tensor, scale_node, zero_node],
                axis=quant_axis)
            graph.replace_input_of_all_nodes(tensor, dequantize_node)
        else:
            new_name = tensor + ".pre"
            if graph.is_graph_output(tensor):
                output_name_from_nodes = graph.output_name_from_nodes()[tensor]
                for in_node in output_name_from_nodes:
                    innode_outputs = in_node.outputs
                    for index in range(len(innode_outputs)):
                        if innode_outputs[index] == tensor:
                            innode_outputs[index] = new_name
                    graph.update_node(in_node, outputs=innode_outputs)
                quantize_node = graph.make_node(
                    'QuantizeLinear',
                    inputs=[new_name, scale_node, zero_node],
                    axis=quant_axis)
                dequantize_node = graph.make_node(
                    'DequantizeLinear',
                    inputs=[quantize_node, scale_node, zero_node],
                    outputs=[tensor],
                    axis=quant_axis)
            else:
                quantize_node = graph.make_node(
                    'QuantizeLinear',
                    inputs=[tensor, scale_node, zero_node],
                    axis=quant_axis)
                dequantize_node = graph.make_node(
                    'DequantizeLinear',
                    inputs=[quantize_node, scale_node, zero_node],
                    axis=quant_axis)
                graph.replace_input_of_all_nodes(tensor, dequantize_node)
    return graph


def new_type_quantize_post_process(graph):
    print("new_type_quantize_post_process")
    # delete all Q and DQ
    graph = remove_all_quantize_ops(graph)

    # merge conv and add
    graph = merge_conv_add(graph)

    # merge conv and bn
    graph = merge_conv_bn(graph)

    # add Q and DQ
    graph = add_q_dq(graph)

    return graph


def save_max_range_file(graph):
    with open("max_range.txt", 'wb') as f:
        for tensor, val in graph.quantize_params_dict.items():
            tensor = tensor + ": " + str(val[2] * 127) + ", " + str(val[3] *
                                                                    127)
            f.write(tensor.encode())


def remove_all_quantize_ops_and_save_max_range_file(graph):
    print("remove_all_quantize_ops_and_save_max_range_file")
    graph = remove_all_quantize_ops(graph)
    # TODO save file
    save_max_range_file(graph)
    return graph


def collect_all_scales(graph):
    node_map = list(graph.ctx.node_map.items())
    for idx in range(len(node_map)):
        name, node = node_map[idx]
        outputs = node.outputs
        for opts in outputs.values():
            for opt in opts:
                if opt in graph.quantize_params_dict:
                    continue
                else:
                    out_threshold = node.attr("out_threshold")
                    if out_threshold is None:
                        continue
                    out_threshold = out_threshold / 127.
                    zero_node = graph.make_node(
                        'Constant', dtype=dtypes.ONNX.INT8, value=0)
                    scale_node = graph.make_node(
                        'Constant',
                        dtype=dtypes.ONNX.FLOAT,
                        value=out_threshold)
                    graph.quantize_params_dict[opt] = [
                        scale_node, zero_node, [out_threshold], [0], -1
                    ]

    return graph


def add_missing_quantize_ops(graph):
    print("add_missing_quantize_ops")
    graph = collect_all_scales(graph)

    # delete all Q and DQ
    graph = remove_all_quantize_ops(graph)

    # merge conv and add
    graph = merge_conv_add(graph)

    # merge conv and bn
    graph = merge_conv_bn(graph)

    # add Q and DQ
    graph = add_q_dq(graph)
    return graph


def delete_redundant_quantize_ops(graph):
    print("delete_redundant_quantize_ops")
    node_map = list(graph.node_map.items())
    for idx in range(len(node_map)):
        name, node = node_map[idx]
        if node.type not in ["QuantizeLinear"]:
            continue
        nodes_to_be_remove = [name]

        inputs = node.inputs[0]
        outputs = node.outputs[0]
        for dequant_idx in range(idx + 1, len(node_map)):
            dq_name, dq_node = node_map[dequant_idx]
            dq_inputs = dq_node.inputs
            dq_outputs = dq_node.outputs
            if outputs in dq_inputs:
                nodes_to_be_remove.append(dq_name)
                outputs = dq_outputs[0]
                break

        if len(graph.input_name_to_nodes()[outputs]) > 1:
            continue

        for op_idx in range(idx + 2, len(node_map)):
            op_name, op_node = node_map[op_idx]
            if len(nodes_to_be_remove) == 0:
                break
            if op_node.type in ["Conv", "MatMul"]:
                continue
            op_inputs = op_node.inputs
            if outputs in op_inputs:
                for input_idx in range(len(op_inputs)):
                    if op_node.inputs[input_idx] == outputs:
                        op_node.inputs[input_idx] = node.inputs[0]
                        op_node.set_inputs(op_node.inputs)
                        graph.update_node(op_node)
                        graph.remove_node_by_name(nodes_to_be_remove[0])
                        graph.remove_node_by_name(nodes_to_be_remove[1])
                        nodes_to_be_remove = []
    return graph


def add_shortcut_quantize_ops(graph):
    print("add_shortcut_quantize_ops_new")
    node_map = list(graph.node_map.items())
    for idx in range(len(node_map)):
        name, node = node_map[idx]
        if node.type not in ['QuantizeLinear']:
            continue
        input_name = node.inputs[0]

        another_nodes = graph.input_name_to_nodes()[input_name]
        if len(another_nodes) == 0:
            continue

        assert input_name in graph.quantize_params_dict, "Can not find quantize param {} in quantize_params_dict".format(
            input_name)
        quantize_nodes = graph.quantize_params_dict[input_name]
        scale_node = quantize_nodes[0]
        zero_node = quantize_nodes[1]
        quant_axis = quantize_nodes[4]
        if quant_axis is None:
            quant_axis = -1
        for another_node in another_nodes:
            if another_node.type in ["QuantizeLinear"]:
                continue
            quantize_node = graph.make_node(
                'QuantizeLinear',
                inputs=[input_name, scale_node, zero_node],
                axis=quant_axis)
            dequantize_node = graph.make_node(
                'DequantizeLinear',
                inputs=[quantize_node, scale_node, zero_node],
                axis=quant_axis)
            inputs = another_node.inputs
            for ipt_idx in range(len(inputs)):
                if inputs[ipt_idx] == input_name:
                    inputs[ipt_idx] = dequantize_node
                    graph.update_node(another_node, inputs=inputs)
    return graph
