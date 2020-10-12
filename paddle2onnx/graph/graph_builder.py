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
from collections import Iterable
import numpy as np
import paddle
from paddle.fluid import core
from paddle.fluid.framework import Variable
import paddle.fluid as fluid
from paddle2onnx.graph import Graph


def build_single_graph(block,
                       parameters=None,
                       input_spec=None,
                       output_spec=None):
    graph = Graph(block.idx, parameters)
    if input_spec is not None:
        for ipt in input_spec:
            layer_name = ipt.name
            attrs = {}
            attrs['shape'] = ipt.shape
            attrs['dtype'] = ipt.dtype
            node = graph.make_node('feed', None, None, attrs, None, layer_name)
            graph.input_nodes.append(node)
    if output_spec is not None:
        for opt in output_spec:
            layer_name = opt.name
            attrs = {}
            attrs['shape'] = opt.shape
            attrs['dtype'] = opt.dtype
            node = graph.make_node('fetch', None, None, attrs, None, layer_name)
            graph.output_nodes.append(node)
    for i, op in enumerate(block.ops):
        if op.type == 'feed':
            layer_name = op.output('Out')[0]
            var = block.var(layer_name)
            attrs = {}
            attrs['shape'] = var.shape
            attrs['dtype'] = var.dtype
            node = graph.make_node(op.type, None, None, attrs, block,
                                   layer_name)
            graph.input_nodes.append(node)
        elif op.type == 'fetch':
            layer_name = op.input('X')[0]
            var = block.var(layer_name)
            attrs = {}
            attrs['shape'] = var.shape
            attrs['dtype'] = var.dtype
            node = graph.make_node(op.type, None, None, attrs, block,
                                   layer_name)
            graph.output_nodes.append(node)
        else:
            inputs = {}
            outputs = {}
            for ipt in op.input_names:
                inputs[ipt] = op.input(ipt)
            for opt in op.output_names:
                outputs[opt] = op.output(opt)
            node = graph.make_node(op.type, inputs, outputs,
                                   op.all_attrs(), block)
            graph.topo_sort.append(node)
    return graph


def build_graph(program, parameters, input_spec=None, output_spec=None):
    # reserve for subgraph 
    graphs = {}

    if len(program.blocks) > 1:
        raise 'Now, paddle export to onnx not support model with sub_block.'

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
