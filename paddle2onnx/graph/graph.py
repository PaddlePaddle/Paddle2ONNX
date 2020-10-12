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
import pickle
import warnings
import inspect
import copy
import collections
import numpy as np


class Node(object):
    def __init__(self, op_type, layer_name, inputs, outputs, attrs, block):
        self.block = block
        self.inputs = inputs
        self.outputs = outputs
        self.type = op_type
        self.attrs = attrs
        self.layer_name = layer_name

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

    def output(self, name, idx=None):
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


class Graph(object):
    def __init__(self, idx, parameters=None):
        self.parameters = parameters
        self.node_map = collections.OrderedDict()
        self.input_nodes = list()
        self.output_nodes = list()
        self.topo_sort = list()
        self.op_type_count = dict()
        self.idx = idx
        self.sub_graphs = list()

    def generate_node_name(self, op_type):
        if op_type in self.op_type_count:
            self.op_type_count[op_type] += 1
        else:
            self.op_type_count[op_type] = 1
        layer_name = op_type + '@block_' + str(self.idx) + '@' + str(
            self.op_type_count[op_type])
        return layer_name

    def make_node(self,
                  op_type,
                  inputs=None,
                  outputs=None,
                  attrs=None,
                  block=None,
                  layer_name=None,
                  **kw):
        if layer_name is None:
            layer_name = self.generate_node_name(op_type)

        if isinstance(inputs, list):
            inputs = [
                ipt.layer_name if isinstance(ipt, Node) else ipt
                for ipt in inputs
            ]
        else:
            assert 'inputs for node must be type: list, but got {}'.format(
                type(inputs))

        if isinstance(outputs, list):
            outputs = [
                opt.layer_name if isinstance(opt, Node) else opt
                for opt in outputs
            ]
        elif outputs == None:
            outputs = [layer_name]
        else:
            assert 'outputs for node must be type: list, but got {}'.format(
                type(outputs))

        if attrs is None:
            attrs = kw
        attrs.update(kw)

        node = Node(op_type, layer_name, inputs, outputs, attrs, block)

        if op_type not in ['feed', 'fetch']:
            self.node_map[node.layer_name] = node
        return node

    def update_node(self,
                    node,
                    op_type,
                    inputs,
                    outputs,
                    attrs=None,
                    block=None,
                    **kw):
        if isinstance(inputs, list):
            inputs = [
                ipt.layer_name if isinstance(ipt, Node) else ipt
                for ipt in inputs
            ]
        else:
            assert 'inputs for node must be type: list, but got {}'.format(
                type(inputs))

        if isinstance(outputs, list):
            outputs = [
                opt.layer_name if isinstance(opt, Node) else opt
                for opt in outputs
            ]
        else:
            assert 'outputs for node must be type: list, but got {}'.format(
                type(outputs))

        node.inputs = inputs
        node.outputs = outputs
        node.type = op_type
        if attrs is None:
            attrs = kw
        attrs.update(kw)
        node.attrs = attrs
        if op_type not in ['feed', 'fetch']:
            self.node_map.pop(node.layer_name)
            self.node_map[node.layer_name] = node
            #self.node_map.move_to_end(node.layer_name)
        return node

    @property
    def get_topo_sort(self):
        # TODO add method to get topo from node_map
        pass

    def get_node(self, name, copy=False):
        if copy:
            node = copy.copy(self.node_map[name])
        else:
            node = self.node_map[name]
        return node

    def remove_node_by_name(self, name):
        if name in self.node_map:
            node = self.node_map.pop(name)
            return node
        assert 'node with name:{} not in graph'.format(name)

    def remove_node(self, node):
        if isinstance(node, Node):
            node = self.remove_node_by_name(node.layer_name)
            return node
        elif isinstance(node, str):
            node = self.remove_node_by_name(node)
            return node
        else:
            assert 'remove node by str or Node, but got type: {}'.format(node)
