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
from paddle2onnx.constant import NodeDomain


class Node(object):
    """
    Args:
        op_type (str): Operator type, such as 'Conv'.
        layer_name (str): Name of node, the name of node in graph is unique. 
        inputs (list|dict): Inputs of node with domain=NodeDomain.ONNX, which stored by list.
            Inputs of node in node with domain=NodeDomain.PADDLE, which stored by key-value format. 
        outputs (list|dict): Outputs of node with domain=NodeDomain.ONNX, which stored by list 
            Outputs of node with domain=NodeDomain.PADDLE, which stored by key-value format 
        attrs (dict): Attributes of node.
        block (paddle.fluid.framework.Block): The block that node belongs to. 
        domain (str):  Domain of node.  

    Returns:
        Node: An Node.
    """

    def __init__(self,
                 op_type,
                 layer_name,
                 inputs,
                 outputs,
                 attrs,
                 block,
                 domain=NodeDomain.PADDLE):
        self.domain = domain
        self.block = block
        self.type = op_type
        self.attrs = attrs
        self.layer_name = layer_name
        self.set_inputs(inputs)
        self.set_outputs(outputs)

    def __hash__(self):
        return hash(self.layer_name)

    def __eq__(self, other):
        if self.layer_name == other.layer_name:
            return True
        return False

    @property
    def input_names(self):
        if isinstance(self.inputs, dict):
            return [name for name in self.inputs.keys()]
        return self.inputs

    @property
    def output_names(self):
        if isinstance(self.inputs, dict):
            return [name for name in self.outputs.keys()]
        return self.outputs

    def input(self, name=None, idx=None):
        if idx is None:
            return self.inputs[name]
        if name is None:
            return self.inputs[idx]
        return self.inputs[name][idx]

    def output(self, name=None, idx=None):
        if idx is None:
            return self.outputs[name]
        if name is None:
            return self.outputs[idx]
        return self.outputs[name][idx]

    def output_shape(self, name=None, idx=None):
        return self.block.var(self.output(name, idx)).shape

    def input_shape(self, name=None, idx=None):
        return self.block.var(self.input(name, idx)).shape

    def input_var(self, name=None, idx=None):
        return self.block.var(self.input(name, idx))

    def attr(self, name):
        return self.attrs[name]

    def set_inputs(self, inputs):
        if isinstance(inputs, list):
            # input of node in onnx, which stored by list 
            self.inputs = [
                ipt.layer_name if isinstance(ipt, Node) else ipt
                for ipt in inputs
            ]
        elif isinstance(inputs, dict):
            # input of node in paddle, which stored by key-value format 
            self.inputs = inputs
        else:
            raise TypeError(
                'Inputs of node must be type: list, dict, but got {}'.format(
                    type(inputs)))

    def set_outputs(self, outputs):
        if isinstance(outputs, list):
            # output of node in onnx, which stored by list 
            self.outputs = [
                opt.layer_name if isinstance(opt, Node) else opt
                for opt in outputs
            ]
        elif isinstance(outputs, dict):
            # output of node in paddle, which stored by key-value format 
            self.outputs = outputs
        else:
            raise TypeError(
                'Outputs of node must be type: list, dict, but got {}'.format(
                    type(outputs)))


class Graph(object):
    """
    Args:
        id (int): the id of graph.
    Returns:
        Graph: A Graph.
    """

    def __init__(self, id):
        self.id = id
        self.parameters = {}
        self.node_map = collections.OrderedDict()
        self.input_nodes = list()
        self.output_nodes = list()
        self.op_type_count = dict()
        self.sub_graphs = list()

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if self.id == other.id:
            return True
        return False

    def __str__(self):
        graph_str = 'graph { \n'
        for node in self.input_nodes:
            graph_str += " input: {} \n".format(node.layer_name)
        for node in self.output_nodes:
            graph_str += " output: {} \n \n".format(node.layer_name)
        for name, node in self.node_map.items():
            if node.domain == NodeDomain.PADDLE:
                node.attrs.pop('op_callstack')
            attrs = ''
            for key, value in node.attrs.items():
                attrs += ', ' + key + '=' + str(value)
            graph_str += "  {} = {}::{}(inputs={}{}) \n".format(
                node.outputs, node.domain, node.type, node.inputs, attrs)
        graph_str += ' }'
        return graph_str

    def set_parameters(self, parameters):
        if isinstance(parameters, dict):
            self.parameters = parameters
        else:
            raise TypeError(
                'parameters of Graph must be type: dict, but got {}'.format(
                    type(parametes)))

    def generate_node_name(self, op_type):
        if op_type in self.op_type_count:
            self.op_type_count[op_type] += 1
        else:
            self.op_type_count[op_type] = 1
        layer_name = op_type + '@block_' + str(self.id) + '@' + str(
            self.op_type_count[op_type])
        return layer_name

    def make_node(self,
                  op_type,
                  inputs=None,
                  outputs=None,
                  attrs=None,
                  block=None,
                  layer_name=None,
                  domain='paddle',
                  **kw):
        if layer_name is None:
            layer_name = self.generate_node_name(op_type)

        if attrs is None:
            attrs = kw
        attrs.update(kw)

        if inputs is None:
            inputs = []
        if outputs is None:
            outputs = [layer_name]

        node = Node(op_type, layer_name, inputs, outputs, attrs, block, domain)

        if op_type not in ['feed', 'fetch']:
            self.node_map[node.layer_name] = node
        return node

    def make_onnx_node(self,
                       op_type,
                       layer_name=None,
                       inputs=None,
                       outputs=None,
                       attrs=None,
                       block=None,
                       **kw):
        domain = NodeDomain.ONNX
        node = self.make_node(
            op_type,
            layer_name=layer_name,
            inputs=inputs,
            outputs=outputs,
            attrs=attrs,
            block=block,
            domain=domain,
            **kw)
        return node

    def update_node(self,
                    node,
                    op_type=None,
                    inputs=None,
                    outputs=None,
                    attrs=None,
                    block=None,
                    move_to_end=True,
                    domain=None,
                    **kw):
        node.type = op_type
        if inputs is not None:
            node.set_inputs(inputs)
        if outputs is not None:
            node.set_outputs(outputs)
        if attrs is None:
            attrs = kw
        attrs.update(kw)
        node.attrs = attrs
        if domain is not None:
            node.domain = domain
        if move_to_end:
            self.node_map.pop(node.layer_name)
            self.node_map[node.layer_name] = node
        return node

    def get_node(self, name, copy=False):
        if name not in self.node_map:
            raise TypeError('Node with name:{} not in graph'.format(name))
        if copy:
            node = copy.copy(self.node_map[name])
        else:
            node = self.node_map[name]
        return node

    def remove_node_by_name(self, name):
        if name in self.node_map:
            node = self.node_map.pop(name)
            return node
        raise TypeError('Node with name:{} not in graph'.format(name))

    def remove_node(self, node):
        if isinstance(node, Node):
            node = self.remove_node_by_name(node.layer_name)
            return node
        elif isinstance(node, str):
            node = self.remove_node_by_name(node)
            return node
        else:
            raise TypeError(
                'Remove node by str or Node, but got type: {}'.format(node))
