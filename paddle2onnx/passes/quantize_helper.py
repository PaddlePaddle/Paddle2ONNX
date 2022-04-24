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


def replace_output_of_all_nodes(graph,
                                old_output_name: str,
                                new_output_name: str):
    for name, node in graph.node_map.items():
        if old_output_name in node.outputs:
            for j in range(len(node.outputs)):
                if node.outputs[j] == old_output_name:
                    node.outputs[j] = new_output_name
                    graph.update_node(node)
                    return


def replace_input_of_all_nodes(graph, old_input_name: str, new_input_name: str):
    for name, node in graph.node_map.items():
        if node.type in ["QuantizeLinear", "DequantizeLinear"]:
            continue
        if old_input_name in node.inputs:
            for j in range(len(node.inputs)):
                if node.inputs[j] == old_input_name:
                    node.inputs[j] = new_input_name
                    graph.update_node(node)


def try_replacing_upstream_output(graph, upstream_output_name, output_name):
    if output_name in graph.quantize_params_dict.keys() and \
        len(graph.input_name_to_nodes()[upstream_output_name]) == 1 and \
        not graph.is_graph_output(upstream_output_name):
        replace_output_of_all_nodes(graph, upstream_output_name, output_name)
        if upstream_output_name in graph.tensor_to_be_quantize:
            graph.tensor_to_be_quantize.remove(upstream_output_name)
        return True
    return False


def remove_all_quantize_ops(graph):
    node_map = list(graph.node_map.items())
    for idx in range(len(node_map)):
        _, q_node = node_map[idx]
        if q_node.type != "QuantizeLinear":
            continue
        nodes_to_be_remove = [q_node]
        q_outputs = q_node.outputs[0]
        q_inputs = q_node.inputs[0]
        input_name_to_nodes_dict = graph.input_name_to_nodes()
        input_name_to_nodes = input_name_to_nodes_dict[q_outputs]
        dq_node = input_name_to_nodes[0]

        assert dq_node.type == "DequantizeLinear", "The output node of QuantizeLinear must be DequantizeLinear"
        nodes_to_be_remove.append(dq_node)
        dq_outputs = dq_node.outputs[0]
        replace_input_of_all_nodes(graph, dq_outputs, q_inputs)

        graph.remove_node(nodes_to_be_remove[0])
        graph.remove_node(nodes_to_be_remove[1])

    return graph


def merge_conv_add(graph):
    node_map = list(graph.node_map.items())
    for idx in range(len(node_map)):
        conv_node_name, conv_node = node_map[idx]
        if conv_node.type != "Conv":
            continue
        conv_outputs = conv_node.outputs[0]
        input_name_to_nodes = graph.input_name_to_nodes()[conv_outputs]
        if len(input_name_to_nodes) > 1 or graph.is_graph_output(conv_outputs):
            continue
        add_node = input_name_to_nodes[0]
        if add_node.type != "Add":
            continue
        add_inputs = add_node.inputs

        bias_node = add_node.inputs[0] if conv_node.outputs[
            0] == add_node.inputs[1] else add_node.inputs[1]
        output_name_from_nodes = graph.output_name_from_nodes()
        # bias node is a constant node, it can not be merge into conv
        if bias_node not in output_name_from_nodes:
            continue
        bias_input_node = output_name_from_nodes[bias_node][0]
        if bias_input_node.type != "Reshape":
            continue
        bias_node = bias_input_node.inputs[0]
        _, bias_weight = mapper_helper.get_param_from_paddle_graph(graph,
                                                                   bias_node)
        if bias_weight is None:
            continue
        _, reshape_tensor = mapper_helper.get_param_from_paddle_graph(
            graph, bias_input_node.inputs[1])
        if reshape_tensor is None:
            continue
        graph.remove_node(bias_input_node)

        topk_data_sort = np.amax(np.abs(bias_weight))
        scale = topk_data_sort / 127.0
        scale_list = scale.tolist()
        if not isinstance(scale_list, list):
            scale_list = [scale_list]
        scale_node = graph.make_node(
            'Constant', dtype=dtypes.ONNX.FLOAT, value=scale_list[0])
        zero_node = graph.make_node('Constant', dtype=dtypes.ONNX.INT8, value=0)
        if bias_node not in graph.quantize_params_dict:
            graph.quantize_params_dict[
                bias_node] = [scale_node, zero_node, scale_list, [0], 0]
        conv_node.inputs.append(bias_node)
        conv_node.outputs = add_node.outputs
        graph.update_node(conv_node)
        graph.remove_node(add_node)
    return graph


def merge_conv_bn(graph):
    node_map = list(graph.node_map.items())
    for idx in range(len(node_map)):
        conv_node_name, conv_node = node_map[idx]
        if conv_node.type != "Conv":
            continue
        conv_outputs = conv_node.outputs[0]
        input_name_to_nodes = graph.input_name_to_nodes()[conv_outputs]
        if len(input_name_to_nodes) > 1 or graph.is_graph_output(conv_outputs):
            continue
        bn_node = input_name_to_nodes[0]
        if bn_node.type != "BatchNormalization":
            continue
        bn_inputs = bn_node.inputs

        conv_weight_node = conv_node.inputs[1]
        weight_params, conv_weight = mapper_helper.get_param_from_paddle_graph(
            graph, conv_weight_node)

        bn_scale_node = bn_node.inputs[1]
        _, bn_scale = mapper_helper.get_param_from_paddle_graph(graph,
                                                                bn_scale_node)
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

        conv_bias = np.zeros((bn_bias.shape[0]))
        conv_bias_node = conv_weight_node + ".merged.bias"
        if len(conv_node.inputs) == 3:
            conv_bias_node = conv_node.inputs[2]
            _, conv_bias = mapper_helper.get_param_from_paddle_graph(
                graph, conv_bias_node)
        graph.only_dequantize.append(conv_bias_node)
        alpha = bn_scale / np.sqrt(bn_var + epsilon)
        new_bias = conv_bias * alpha + (bn_bias - bn_mean * alpha)
        new_weight = conv_weight * np.expand_dims(alpha, axis=[1, 2, 3])
        weight_params["data"] = new_weight
        graph.update_parameters(conv_weight_node, weight_params)

        # update weight scale
        quantize_params = graph.quantize_params_dict[conv_weight_node]
        scale_node = quantize_params[0]
        zero_node = quantize_params[1]
        graph.remove_node_by_name(scale_node)
        graph.remove_node_by_name(zero_node)
        scale_list = quantize_params[2]
        quant_axis = quantize_params[4]
        if len(scale_list) == 1:
            scale_list, zero_list = mapper_helper.quantize_weight(new_weight)
            scale = np.array(scale_list)
            zero_node = graph.make_node(
                'Constant', dtype=dtypes.ONNX.INT8, value=zero_list)
            scale_node = graph.make_node(
                'Constant', dtype=dtypes.ONNX.FLOAT, value=scale_list)
            graph.quantize_params_dict[conv_weight_node] = [
                scale_node, zero_node, scale_list, [0], quant_axis
            ]
        else:
            scale_list, zero_list = mapper_helper.quantize_weight_per_channel(
                new_weight, quant_axis)
            scale = np.array(scale_list)
            zero_node = graph.make_node(
                'Constant', dtype=dtypes.ONNX.INT8, value=zero_list)
            scale_node = graph.make_node(
                'Constant', dtype=dtypes.ONNX.FLOAT, value=scale_list)
            graph.quantize_params_dict[conv_weight_node] = [
                scale_node, zero_node, scale_list, zero_list, quant_axis
            ]

        # bias scale and bias
        scale = np.squeeze(scale *
                           graph.quantize_params_dict[conv_node.inputs[0]][2])
        scale_list = scale.tolist()
        if not isinstance(scale_list, list):
            scale_list = [scale_list]
        if len(scale_list) == 1:
            scale_node = graph.make_node(
                'Constant', dtype=dtypes.ONNX.FLOAT, value=scale_list[0])
            zero_node = graph.make_node(
                'Constant', dtype=dtypes.ONNX.INT32, value=0)
        else:
            scale_node = graph.make_node(
                'Constant', dtype=dtypes.ONNX.FLOAT, value=scale_list)
            zero_node = graph.make_node(
                'Constant',
                dtype=dtypes.ONNX.INT32,
                value=[0] * len(scale_list))

        conv_node.inputs.append(conv_bias_node)
        conv_node.outputs = bn_node.outputs
        graph.update_node(conv_node)
        graph.remove_node(bn_node)
        if graph.quantize_model_mode == "float":
            return graph
        graph.quantize_params_dict[
            conv_bias_node] = [scale_node, zero_node, scale_list, [0], -1]
    return graph


def can_be_quantize(tensor_names: list, graph):
    for tensor_name in tensor_names:
        if tensor_name not in graph.quantize_params_dict:
            logging.warning("[Quantize] Find unquantize tensor: {}".format(
                tensor_name))
            return False
    return True


def add_q_dq(graph):
    node_map = list(graph.node_map.items())
    for idx in range(len(node_map)):
        name, node = node_map[idx]
        if node.type in ["Relu"]:
            if not can_be_quantize([node.inputs[0], node.outputs[0]], graph):
                continue
            graph.make_node(
                "LeakyRelu",
                inputs=node.inputs,
                outputs=node.outputs,
                alpha=0.0)
            graph.remove_node(node)
            graph.tensor_to_be_quantize.append(node.inputs[0])
            graph.tensor_to_be_quantize.append(node.outputs[0])
            if node.inputs[
                    0] not in graph.quantize_params_dict and node.outputs[
                        0] in graph.quantize_params_dict:
                graph.quantize_params_dict[node.inputs[
                    0]] = graph.quantize_params_dict[node.outputs[0]]
        if node.type in [
                "Reshape", "Transpose", "Squeeze", "Unsqueeze", "AveragePool"
        ]:
            if not can_be_quantize([node.inputs[0], node.outputs[0]], graph):
                continue
            graph.tensor_to_be_quantize.append(node.inputs[0])
            graph.tensor_to_be_quantize.append(node.outputs[0])

        if node.type in ["Conv"]:
            tensor_names = [node.inputs[0], node.inputs[1], node.outputs[0]]
            if len(node.inputs) == 3:
                tensor_names.append(node.inputs[2])
            if not can_be_quantize(tensor_names, graph):
                continue
            for tensor_name in tensor_names:
                graph.tensor_to_be_quantize.append(tensor_name)

        if node.type in ["Resize", "MaxPool"]:
            if not can_be_quantize([node.inputs[0], node.outputs[0]], graph):
                continue
            graph.tensor_to_be_quantize.append(node.inputs[0])
            graph.tensor_to_be_quantize.append(node.outputs[0])

        if node.type in ["Concat", "MatMul"]:
            tensors_to_quantize = node.inputs
            if not can_be_quantize(tensors_to_quantize, graph):
                continue
            for tensor_name in tensors_to_quantize:
                graph.tensor_to_be_quantize.append(tensor_name)

        if node.type in [
                "ArgMax", "Mul", "LeakyRelu", "Sigmoid", "GlobalAveragePool",
                "Split", "Pad", "Add"
        ]:
            tensors_to_quantize = []
            for tensor_name in itertools.chain(node.inputs, node.outputs):
                tensors_to_quantize.append(tensor_name)
            if not can_be_quantize(tensors_to_quantize, graph):
                continue
            for tensor_name in tensors_to_quantize:
                graph.tensor_to_be_quantize.append(tensor_name)

    graph.tensor_to_be_quantize = list(set(graph.tensor_to_be_quantize))
    for tensor in graph.tensor_to_be_quantize:
        if tensor not in graph.quantize_params_dict:
            logging.warning("[Quantize] Find unquantize tensor: {}".format(
                tensor))
            continue
        quantize_params = graph.quantize_params_dict[tensor]
        assert len(
            quantize_params
        ) == 5, "The len of quantize_params must be 5, but get length is: {}".format(
            len(quantize_params))
        scale_node = quantize_params[0]
        zero_node = quantize_params[1]
        quant_axis = quantize_params[4]
        if quant_axis is None:
            quant_axis = -1

        if tensor in graph.only_dequantize:
            params, weight = mapper_helper.get_param_from_paddle_graph(graph,
                                                                       tensor)
            scale = quantize_params[2][0]
            new_weight = weight / scale
            params['shape'] = [new_weight.shape[0]]
            params["data"] = np.round(new_weight).astype("int32")
            params['dtype'] = paddle.int8
            graph.update_parameters(tensor, params)
            dequantize_node = graph.make_node(
                'DequantizeLinear',
                inputs=[tensor, scale_node, zero_node],
                axis=quant_axis)
            replace_input_of_all_nodes(graph, tensor, dequantize_node)
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
                replace_input_of_all_nodes(graph, tensor, dequantize_node)
    return graph


def new_type_quantize_post_process(graph):
    # delete all Q and DQ
    graph = remove_all_quantize_ops(graph)

    # merge conv and add
    graph = merge_conv_add(graph)

    # merge conv and bn
    graph = merge_conv_bn(graph)

    # add Q and DQ
    graph = add_q_dq(graph)

    return graph


def remove_all_quantize_ops_and_save_max_range_file(graph):
    graph = remove_all_quantize_ops(graph)
    scales = dict()
    for tensor, val in graph.quantize_params_dict.items():
        scales[tensor] = [val[2] * 127, val[3] * 127]
    graph.quantize_params_dict = scales
    return graph


def collect_all_scales(graph):
    input_name_to_nodes_dict = graph.input_name_to_nodes()
    node_map = list(graph.ctx.node_map.items())
    for idx in range(len(node_map)):
        name, node = node_map[idx]
        outputs = node.outputs
        for key, opts in outputs.items():
            if key not in ["Y", "Out", "Output"]:
                continue
            for opt in opts:
                input_name = opt
                if opt in graph.static_quantize_pre_convert_dict.keys():
                    input_name = graph.static_quantize_pre_convert_dict[opt]
                if input_name in graph.quantize_params_dict:
                    continue
                if input_name not in input_name_to_nodes_dict:
                    continue
                out_threshold = node.attr("out_threshold")
                if out_threshold is None:
                    continue
                out_threshold = out_threshold / 127
                zero_node = graph.make_node(
                    'Constant', dtype=dtypes.ONNX.INT8, value=0)
                scale_node = graph.make_node(
                    'Constant', dtype=dtypes.ONNX.FLOAT, value=out_threshold)
                graph.quantize_params_dict[input_name] = [
                    scale_node, zero_node, [out_threshold], [0], -1
                ]
    return graph


# onnxruntime deploy for static and dynamic
def add_missing_quantize_ops(graph):
    graph = collect_all_scales(graph)

    # delete all Q and DQ
    graph = remove_all_quantize_ops(graph)

    # merge conv and add
    graph = merge_conv_add(graph)

    # merge conv and bn
    graph = merge_conv_bn(graph)

    graph = add_q_dq(graph)
    return graph


def delete_redundant_quantize_ops(graph):
    node_map = list(graph.node_map.items())
    for idx in range(len(node_map)):
        _, q_node = node_map[idx]
        if q_node.type != "QuantizeLinear":
            continue
        nodes_to_be_remove = [q_node]
        q_inputs = q_node.inputs[0]
        q_outputs = q_node.outputs[0]
        input_name_to_nodes_dict = graph.input_name_to_nodes()
        input_name_to_nodes = input_name_to_nodes_dict[q_outputs]
        dq_node = input_name_to_nodes[0]
        assert dq_node.type == "DequantizeLinear", "The output node of QuantizeLinear must be DequantizeLinear"
        nodes_to_be_remove.append(dq_node)
        dq_outputs = dq_node.outputs[0]
        op_nodes = input_name_to_nodes_dict[dq_outputs]
        op_types = [op_node.type for op_node in op_nodes]
        if "Conv" in op_types or "MatMul" in op_types:
            continue
        graph.remove_node(nodes_to_be_remove[0])
        graph.remove_node(nodes_to_be_remove[1])
        replace_input_of_all_nodes(graph, dq_outputs, q_inputs)
    return graph


def add_shortcut_quantize_ops(graph):
    node_map = list(graph.node_map.items())
    for idx in range(len(node_map)):
        name, node = node_map[idx]
        if node.type != 'QuantizeLinear':
            continue
        input_name = node.inputs[0]

        another_nodes = graph.input_name_to_nodes()[input_name]
        if len(another_nodes) == 0:
            continue

        assert input_name in graph.quantize_params_dict, "Can not find quantize param {} in quantize_params_dict".format(
            input_name)
        quantize_params = graph.quantize_params_dict[input_name]
        scale_node = quantize_params[0]
        zero_node = quantize_params[1]
        quant_axis = quantize_params[4]
        if quant_axis is None:
            quant_axis = -1
        for another_node in another_nodes:
            if another_node.type == "QuantizeLinear":
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
