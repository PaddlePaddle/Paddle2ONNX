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

from collections import Iterable
import paddle
from paddle.fluid.framework import Operator
from paddle2onnx.graph import Graph


def add_input_node(graph, input_spec=None, op=None, block=None):
    if isinstance(input_spec, Iterable):
        for ipt in input:
            if isinstance(ipt, paddle.static.InputSpec):
                layer_name = ipt.name
                attrs = {}
                attrs['shape'] = ipt.shape
                attrs['dtype'] = ipt.dtype
                node = graph.make_node('feed', [], [layer_name], attrs, None,
                                       layer_name)
                graph.input_nodes.append(node)
    if isinstance(op, Operator):
        layer_name = op.output('Out')[0]
        var = block.var(layer_name)
        attrs = {}
        attrs['shape'] = var.shape
        attrs['dtype'] = var.dtype
        node = graph.make_node('feed', [], [layer_name], attrs, block,
                               layer_name)
        graph.input_nodes.append(node)
    return graph


def add_output_node(graph, output_spec=None, op=None, block=None):
    if isinstance(output_spec, Iterable):
        for opt in output_spec:
            layer_name = opt.name
            attrs = {}
            attrs['shape'] = opt.shape
            attrs['dtype'] = opt.dtype
            node = graph.make_node('fetch', [layer_name], [], attrs, None,
                                   layer_name)
            graph.output_nodes.append(node)
    if isinstance(op, Operator):
        layer_name = op.input('X')[0]
        var = block.var(layer_name)
        attrs = {}
        attrs['shape'] = var.shape
        attrs['dtype'] = var.dtype
        node = graph.make_node(op.type, [layer_name], [], attrs, block,
                               layer_name)
        graph.output_nodes.append(node)
    return graph


def build_single_graph(block,
                       parameters=None,
                       input_spec=None,
                       output_spec=None):
    graph = Graph(block.idx)

    if parameters is not None:
        graph.set_parameters(parameters)

    graph = add_input_node(graph, input_spec=input_spec)
    graph = add_output_node(graph, output_spec=output_spec)

    for i, op in enumerate(block.ops):
        if op.type == 'feed':
            graph = add_input_node(graph, op=op, block=block)
        elif op.type == 'fetch':
            graph = add_output_node(graph, op=op, block=block)
        else:
            inputs = {}
            outputs = {}
            for ipt in op.input_names:
                inputs[ipt] = op.input(ipt)
            for opt in op.output_names:
                outputs[opt] = op.output(opt)
            node = graph.make_node(op.type, inputs, outputs,
                                   op.all_attrs(), block)
    return graph


def build_graph(program, parameters, input_spec=None, output_spec=None):

    # reserve for subgraph 
    graphs = {}

    if len(program.blocks) > 1:
        raise 'Now, paddle export to onnx not support model with multiple blocks.'

    # TODO support parse parameters for model with multiple blocks 
    for block in program.blocks:
        if block.idx == 0:
            graph = build_single_graph(block, parameters, input_spec,
                                       output_spec)
        else:
            graph = build_single_graph(block)

        graphs[block.idx] = graph

        if block.parent_idx in graphs:
            graphs[block.parent_idx].sub_graphs.append(graph)
    return graphs[0]
