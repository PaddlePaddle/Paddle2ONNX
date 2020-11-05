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

import os
import copy
import collections
import numpy as np
from paddle2onnx.graph import Node, Graph
from paddle2onnx.constant import NodeDomain, PRODUCER, dtypes
from paddle2onnx.op_mapper import OpMapper
from paddle2onnx.onnx_helper import helper
from paddle2onnx.utils import check_model, logging
from paddle2onnx.constant.op_mapping_status import *


class ONNXNode(Node):
    def __init__(self, op_type, inputs, outputs, attrs, layer_name):
        """
        Initialize onnx graph.

        Args:
            self: (todo): write your description
            op_type: (todo): write your description
            inputs: (list): write your description
            outputs: (str): write your description
            attrs: (dict): write your description
            layer_name: (str): write your description
        """
        super(ONNXNode, self).__init__(op_type, inputs, outputs, attrs,
                                       layer_name, NodeDomain.ONNX)
        self.onnx_node = self.make_onnx_node()

    def make_onnx_constant_node(self):
        """
        Create a tensorfluent node.

        Args:
            self: (todo): write your description
        """
        dtype = self.attr('dtype')
        value = self.attr('value')
        if isinstance(value, list):
            dims = (len(value), )
        elif value is None:
            dims = ()
            value = []
        else:
            dims = ()
            value = [value]

        if 'dims' in self.attrs:
            dims = self.attrs['dims']

        tensor = helper.make_tensor(
            name=self.layer_name, data_type=dtype, dims=dims, vals=value)

        onnx_node = helper.make_node(
            self.type, inputs=self.inputs, outputs=self.outputs, value=tensor)

        return onnx_node

    def make_onnx_node(self):
        """
        Create a node for onnxnode

        Args:
            self: (todo): write your description
        """
        if self.type in ['Constant', 'ConstantOfShape']:
            onnx_node = self.make_onnx_constant_node()
        else:
            onnx_node = helper.make_node(
                self.type,
                inputs=self.inputs,
                outputs=self.outputs,
                name=self.layer_name,
                **self.attrs)
        return onnx_node


class ONNXGraph(Graph):
    op_mapping_status = {
        OP_MAPPING_NO_REGISTER: [],
        OP_MAPPING_NO_VERSION: [],
        OP_MAPPING_SUCCESSED: [],
    }

    def __init__(self, paddle_graph, opset_version, block=None):
        """
        Initialize the graph

        Args:
            self: (todo): write your description
            paddle_graph: (todo): write your description
            opset_version: (str): write your description
            block: (todo): write your description
        """
        super(ONNXGraph, self).__init__()
        self.opset_version = opset_version
        self.ctx = paddle_graph

    def make_node(self,
                  op_type,
                  inputs=None,
                  outputs=None,
                  attrs=None,
                  layer_name=None,
                  **kw):
        """
        Creates a node from the given type.

        Args:
            self: (todo): write your description
            op_type: (str): write your description
            inputs: (todo): write your description
            outputs: (todo): write your description
            attrs: (dict): write your description
            layer_name: (str): write your description
            kw: (todo): write your description
        """
        if layer_name is None:
            layer_name = self.generate_node_name(op_type)

        if attrs is None:
            attrs = kw
        attrs.update(kw)

        if inputs is None:
            inputs = []
        if outputs is None:
            outputs = [layer_name]
        node = ONNXNode(op_type, inputs, outputs, attrs, layer_name)
        self.insert_node(node)
        return node

    def make_value_info(self, name, shape, dtype):
        """
        Create a tensorvalueinfo object.

        Args:
            self: (todo): write your description
            name: (str): write your description
            shape: (int): write your description
            dtype: (todo): write your description
        """
        tensor_info = helper.make_tensor_value_info(
            name=name,
            shape=shape,
            elem_type=dtypes.DTYPE_PADDLE_ONNX_MAP[dtype])
        return tensor_info

    def add_input_node(self, name, shape, dtype):
        """
        Add an input node.

        Args:
            self: (todo): write your description
            name: (str): write your description
            shape: (int): write your description
            dtype: (todo): write your description
        """
        vi = self.make_value_info(name, shape, dtype)
        self.input_nodes.append(vi)

    def add_output_node(self, name, shape, dtype):
        """
        Add an output node.

        Args:
            self: (todo): write your description
            name: (str): write your description
            shape: (int): write your description
            dtype: (todo): write your description
        """
        vi = self.make_value_info(name, shape, dtype)
        self.output_nodes.append(vi)

    def export_proto(self, enable_onnx_checker=False):
        """
        Export the graph asnx graph

        Args:
            self: (todo): write your description
            enable_onnx_checker: (bool): write your description
        """
        op_nodes = [node.onnx_node for node in self.node_map.values()]
        weight_nodes = [node for node in self.parameters.values()]
        onnx_graph = helper.make_graph(
            nodes=weight_nodes + op_nodes,
            name='paddle-onnx',
            initializer=[],
            inputs=self.input_nodes,
            outputs=self.output_nodes)

        opset_imports = [helper.make_opsetid("", self.opset_version)]
        onnx_proto = helper.make_model(
            onnx_graph, producer_name=PRODUCER, opset_imports=opset_imports)

        if enable_onnx_checker:
            check_model(onnx_proto)
        return onnx_proto

    def check_op_mapping_status(self):
        """
        Check if the op_mapping of op_status of the op_types.

        Args:
            self: (todo): write your description
        """
        if len(self.op_mapping_status[OP_MAPPING_NO_REGISTER]) > 0:
            unsupported_op_types = set([
                node.type
                for node in self.op_mapping_status[OP_MAPPING_NO_REGISTER]
            ])
            error_info = "\nThere's {} ops are not supported yet\n".format(
                len(unsupported_op_types))
            for op_type in unsupported_op_types:
                error_info += "=========== {} ===========\n".format(op_type)
            raise NotImplementedError(error_info)

        if len(self.op_mapping_status[OP_MAPPING_NO_VERSION]) > 0:
            unsupported_op_types = set([
                node.type
                for node in self.op_mapping_status[OP_MAPPING_NO_VERSION]
            ])
            error_info = "\nThere's {} ops are not supported in opset_version {}, please try other opset versions\n".format(
                len(unsupported_op_types), self.opset_version)

            for op_type in unsupported_op_types:
                error_info += "=========== {} ===========\n".format(op_type)
            raise NotImplementedError(error_info)

    @staticmethod
    def build(paddle_graph, opset_version, verbose=False):
        """
        Build onnx graph from onnx graph

        Args:
            paddle_graph: (todo): write your description
            opset_version: (str): write your description
            verbose: (bool): write your description
        """
        onnx_graph = ONNXGraph(paddle_graph, opset_version=opset_version)

        # build input nodes
        for ipt in paddle_graph.input_nodes:
            vi = onnx_graph.add_input_node(ipt.layer_name,
                                           ipt.attr('shape'), ipt.attr('dtype'))

        # build output nodes
        for opt in paddle_graph.output_nodes:
            onnx_graph.add_output_node(opt.layer_name,
                                       opt.attr('shape'), opt.attr('dtype'))

        # build weight nodes
        for name, param in paddle_graph.parameters.items():
            weight = param['data']
            if weight is not np.ndarray:
                weight = np.array(weight)
            tensor = helper.make_tensor(
                name=name,
                dims=param['shape'],
                data_type=dtypes.DTYPE_PADDLE_ONNX_MAP[param['dtype']],
                vals=weight.flatten().tolist())
            node = helper.make_node(
                'Constant', inputs=[], outputs=[name], value=tensor)
            onnx_graph.parameters[name] = node

        # build op nodes
        for name, node in list(paddle_graph.node_map.items()):
            status = OpMapper.mapping(onnx_graph, node)
            onnx_graph.op_mapping_status[status].append(node)

        onnx_graph.check_op_mapping_status()

        return onnx_graph
