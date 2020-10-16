# Copyright (c) 2020  PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np
import onnx
import copy
from paddle2onnx.constant import dtypes
from paddle2onnx.op_mapper import OpMapper
from paddle2onnx.constant.op_mapping_status import *


def make_value_info(name, shape, dtype):
    tensor_info = onnx.helper.make_tensor_value_info(
        name=name, shape=shape, elem_type=dtypes.DTYPE_PADDLE_ONNX_MAP[dtype])
    return tensor_info


def inputs_to_onnx(inputs=None):
    input_nodes = []
    for ipt in inputs:
        vi = make_value_info(ipt.layer_name,
                             ipt.attr('shape'), ipt.attr('dtype'))
        input_nodes.append(vi)
    return input_nodes


def outputs_to_onnx(outputs=None):
    output_nodes = []
    for opt in outputs:
        vi = make_value_info(opt.layer_name,
                             opt.attr('shape'), opt.attr('dtype'))
        output_nodes.append(vi)
    return output_nodes


def weights_to_onnx(parameters=None):
    nodes = list()
    if parameters is None:
        return nodes
    for name, param in parameters.items():
        weight = param['data']
        if weight is not np.ndarray:
            weight = np.array(weight)
        tensor = onnx.helper.make_tensor(
            name=name,
            dims=param['shape'],
            data_type=dtypes.DTYPE_PADDLE_ONNX_MAP[param['dtype']],
            vals=weight.flatten().tolist())
        node = onnx.helper.make_node(
            'Constant', inputs=[], outputs=[name], value=tensor)
        nodes.append(node)
    return nodes


def make_onnx_constant_node(node):
    dtype = node.attr('dtype')
    value = node.attr('value')
    if isinstance(value, list):
        dims = (len(value), )
    elif value is None:
        dims = ()
        value = []
    else:
        dims = ()
        value = [value]

    if 'dims' in node.attrs:
        dims = node.attrs['dims']

    tensor = onnx.helper.make_tensor(
        name=node.layer_name, data_type=dtype, dims=dims, vals=value)
    onnx_node = onnx.helper.make_node(
        'Constant', inputs=[], outputs=node.outputs, value=tensor)
    return onnx_node


def make_onnx_node(node):
    if node.type == 'Constant':
        return make_onnx_constant_node(node)
    else:
        onnx_node = onnx.helper.make_node(
            node.type, inputs=node.inputs, outputs=node.outputs, **node.attrs)
        return onnx_node


def check_op_mapping_status(mapping_status, opset_version):
    if len(mapping_status[OP_MAPPING_NO_REGISTER]) > 0:
        unsupported_op_types = set(
            [node.type for node in mapping_status[OP_MAPPING_NO_REGISTER]])
        error_info = "\nThere's {} ops are not supported yet\n".format(
            len(unsupported_op_types))
        for op_type in unsupported_op_types:
            error_info += "=========== {} ===========\n".format(op_type)
        raise NotImplementedError(error_info)

    if len(mapping_status[OP_MAPPING_NO_VERSION]) > 0:
        unsupported_op_types = set(
            [node.type for node in mapping_status[OP_MAPPING_NO_VERSION]])
        error_info = "\nThere's {} ops are not supported in opset_version {}, please try lower opset version\n".format(
            len(unsupported_op_types), opset_version)

        for op_type in unsupported_op_types:
            error_info += "=========== {} ===========\n".format(op_type)
        raise NotImplementedError(error_info)


def nodes_to_onnx(graph, opset_version, verbose=False):
    mapping_status = {
        OP_MAPPING_NO_REGISTER: [],
        OP_MAPPING_NO_VERSION: [],
        OP_MAPPING_SUCCESSED: [],
        OP_MAPPING_FAILED: [],
    }

    for name, node in list(graph.node_map.items()):
        status = OpMapper.mapping(graph, node, opset_version)
        mapping_status[status].append(node)

    check_op_mapping_status(mapping_status, opset_version)

    onnx_nodes = []

    if verbose:
        print(graph)
    for name, node in list(graph.node_map.items()):
        onnx_node = make_onnx_node(node)
        onnx_nodes.append(onnx_node)

    return onnx_nodes


def graph_to_onnx(graph, opset_version, verbose=False):
    onnx_graphs = {}
    graph = copy.copy(graph)
    input_nodes = inputs_to_onnx(graph.input_nodes)
    output_nodes = outputs_to_onnx(graph.output_nodes)
    weight_nodes = weights_to_onnx(graph.parameters)

    op_nodes = nodes_to_onnx(graph, opset_version, verbose=verbose)

    onnx_graph = onnx.helper.make_graph(
        nodes=weight_nodes + op_nodes,
        name='paddle-onnx',
        initializer=[],
        inputs=input_nodes,
        outputs=output_nodes)

    onnx_graphs[graph.id] = onnx_graph

    for graph in graph.sub_graphs:
        return onnx_graphs.update(graph_to_onnx(graph, opset_version))

    return onnx_graphs
