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
from paddle.fluid import dygraph
from paddle.fluid.framework import Operator
from paddle2onnx.graph import Node, Graph
from paddle2onnx.constant import NodeDomain


class PaddleNode(Node):
    def __init__(self, paddle_op, inputs, outputs, attrs, layer_name, block):
        super(PaddleNode, self).__init__(paddle_op.type, inputs, outputs, attrs,
                                         layer_name, NodeDomain.PADDLE)
        self.paddle_op = paddle_op
        self.block = block

    @property
    def input_names(self):
        return [name for name in self.inputs.keys()]

    @property
    def output_names(self):
        return [name for name in self.outputs.keys()]

    def input(self, name, idx=None):
        if idx is None:
            return self.inputs[name]
        return self.inputs[name][idx]

    def output(self, name, idx=None):
        if idx is None:
            return self.outputs[name]
        return self.outputs[name][idx]

    def output_shape(self, name, idx):
        return self.block.var(self.output(name, idx)).shape

    def input_shape(self, name, idx):
        return self.block.var(self.input(name, idx)).shape

    def input_var(self, name, idx):
        return self.block.var(self.input(name, idx))

    def attr(self, name):
        if name in self.attrs:
            return self.attrs[name]
        return None

    def set_inputs(self, inputs):
        if isinstance(inputs, dict):
            # input of node in paddle, which stored by dict 
            self.inputs = inputs
        else:
            raise TypeError('Inputs of node must be type: dict, but got {}'.
                            format(type(inputs)))

    def set_outputs(self, outputs):
        if isinstance(outputs, dict):
            # output of node in paddle, which stored by dict 
            self.outputs = outputs
        else:
            raise TypeError('Outputs of node must be type: dict, but got {}'.
                            format(type(outputs)))


class PaddleGraph(Graph):
    def __init__(self, program, parameters, input_spec=None, output_spec=None):
        super(PaddleGraph, self).__init__()
        self.build_graph(program, parameters, input_spec, output_spec)

    def make_node(self,
                  op,
                  inputs=None,
                  outputs=None,
                  attrs=None,
                  block=None,
                  layer_name=None,
                  **kw):
        if layer_name is None:
            layer_name = self.generate_node_name(op.type)

        if attrs is None:
            attrs = kw
        attrs.update(kw)

        if inputs is None:
            inputs = {}
        if outputs is None:
            outputs = {'Out': layer_name}
        node = PaddleNode(op, inputs, outputs, attrs, layer_name, block)
        self.insert_node(node)
        return node

    def add_input_node(self, input_spec=None, op=None, block=None):
        if isinstance(input_spec, collections.Iterable):
            for ipt in input:
                if isinstance(ipt, paddle.static.InputSpec):
                    layer_name = ipt.name
                    attrs = {}
                    attrs['shape'] = ipt.shape
                    attrs['dtype'] = ipt.dtype
                    node = Node('feed', [], [layer_name], attrs, layer_name)
                    self.input_nodes.append(node)
        if isinstance(op, Operator):
            layer_name = op.output('Out')[0]
            var = block.var(layer_name)
            attrs = {}
            attrs['shape'] = var.shape
            attrs['dtype'] = var.dtype
            node = Node('feed', [], [layer_name], attrs, layer_name)
            self.input_nodes.append(node)

    def add_output_node(self, output_spec=None, op=None, block=None):
        if isinstance(output_spec, collections.Iterable):
            for opt in output_spec:
                layer_name = opt.name
                attrs = {}
                attrs['shape'] = opt.shape
                attrs['dtype'] = opt.dtype
                node = Node('fetch', [layer_name], [], attrs, layer_name)
                self.output_nodes.append(node)
        if isinstance(op, Operator):
            layer_name = op.input('X')[0]
            var = block.var(layer_name)
            attrs = {}
            attrs['shape'] = var.shape
            attrs['dtype'] = var.dtype
            node = Node(op.type, [layer_name], [], attrs, layer_name)
            self.output_nodes.append(node)

    def get_adjacency_map(self):
        adjacency_map = {}
        for layer_name, current_node in self.node_map.items():
            inputs = current_node.inputs.values()
            inputs = [x for j in inputs for x in j]
            for ipt in inputs:
                for layer_name, node in self.node_map.items():
                    if current_node == node:
                        continue
                    outputs = node.outputs.values()
                    outputs = [x for j in outputs for x in j]
                    if ipt in outputs:
                        if node not in adjacency_map:
                            adjacency_map[node] = set([current_node])
                        else:
                            adjacency_map[node].add(current_node)
        return adjacency_map

    def build_graph(self,
                    program,
                    parameters,
                    input_spec=None,
                    output_spec=None):
        self.program = program
        self.set_parameters(parameters)
        self.add_input_node(input_spec=input_spec)
        self.add_output_node(output_spec=output_spec)
        for block in program.blocks:
            for i, op in enumerate(block.ops):
                if op.type == 'feed':
                    self.add_input_node(op=op, block=block)
                elif op.type == 'fetch':
                    self.add_output_node(op=op, block=block)
                else:
                    inputs = {}
                    outputs = {}
                    for ipt in op.input_names:
                        inputs[ipt] = op.input(ipt)
                    for opt in op.output_names:
                        outputs[opt] = op.output(opt)
                    node = self.make_node(op, inputs, outputs,
                                          op.all_attrs(), block)

    @staticmethod
    def build_from_program(program, feed=None, fetch=None, scope=None):
        parameters_dict = {}
        vars = program.global_block().vars
        for name in vars:
            var = program.global_block().var(name)
            if name.endswith('feed') or name.endswith('fetch'):
                continue
            if not var.persistable:
                continue
            parameters_dict[name] = {
                'data': np.array(scope.var(name).get_tensor()),
                'dtype': var.dtype,
                'shape': var.shape
            }
        graph = PaddleGraph(program, parameters_dict)
        return graph

    @staticmethod
    def build_from_dygraph(layer, input_spec=None, output_spec=None):
        from paddle.nn import Layer
        from paddle.fluid import core
        from paddle.fluid.framework import Variable
        from paddle2onnx.graph import dygraph_helper as dg_helper
        if isinstance(layer, dygraph.TranslatedLayer):
            program = layer.program()
            parameters_dict = {}
            pruned_vars = program.global_block().vars
            for param in layer.parameters():
                if param.name.endswith('feed') or param.name.endswith('fetch'):
                    continue
                if not param.persistable:
                    continue
                if param.name in pruned_vars:
                    parameters_dict[param.name] = {
                        'data': np.array(param.value().get_tensor()),
                        'dtype': param.dtype,
                        'shape': param.shape
                    }
            graph = PaddleGraph(program, parameters_dict,
                                layer._input_spec(), layer._output_spec())
            return graph
        elif isinstance(layer, Layer):
            program = dg_helper.get_program(layer, input_spec, output_spec)
            parameters_dict = {}
            pruned_vars = program.global_block().vars
            for param in layer.parameters():
                if param.name.endswith('feed') or param.name.endswith('fetch'):
                    continue
                if not param.persistable:
                    continue
                if param.name in pruned_vars:
                    parameters_dict[param.name] = {
                        'data': np.array(param.value().get_tensor()),
                        'dtype': param.dtype,
                        'shape': param.shape
                    }
            graph = PaddleGraph(program, parameters_dict)
            return graph
        else:
            raise TypeError(
                "The input Layer should be 'Layer' or 'TranslatedLayer', but received  type is %s."
                % type(layer))
