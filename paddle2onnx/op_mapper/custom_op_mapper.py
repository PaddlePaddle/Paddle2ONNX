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
import paddle
from paddle.fluid import layers
from paddle import fluid
from paddle2onnx.graph import graph_helper, PaddleGraph

REGISTER_CUSTOM_OP = {}


def regist_custom_paddle_op(paddle_op, custom_op):
    if not isinstance(paddle_op, list):
        paddle_op = [paddle_op]
        for op in paddle_op:
            if op not in REGISTER_CUSTOM_OP:
                REGISTER_CUSTOM_OP[op] = custom_op


class Inputs():
    def __init__(self, inputs):
        self.inputs = inputs

    def input(self, name, idx=None):
        if name not in self.inputs:
            return None
        if idx is None:
            return self.inputs[name]
        if len(self.inputs) <= idx:
            return None
        return self.inputs[name][idx]


class CustomOp():
    def __init__(self, node):
        self.main_program = paddle.static.Program()
        self.startup_program = paddle.static.Program()
        self.inputs = self.create_place_holder(node)
        self.node = node

    def create_place_holder(self, node):
        place_holders = {}
        with paddle.static.program_guard(self.main_program,
                                         self.startup_program):
            for arg_name, idxs in node.inputs.items():
                place_holders[arg_name] = []
                for idx in range(len(idxs)):
                    shape = node.input_shape(arg_name, idx)
                    dtype = node.input_dtype(arg_name, idx)
                    name = node.input(arg_name, idx)
                    data = paddle.static.data(
                        name=name, shape=shape, dtype=dtype)
                    place_holders[arg_name].append(data)
        return Inputs(place_holders)

    def rename_output_node(self, graph, old_name, new_name):
        output_idx = None
        for idx in range(len(graph.output_nodes)):
            if graph.output_nodes[idx].layer_name == old_name:
                output_idx = idx
                break
        graph.output_nodes[output_idx].layer_name = new_name
        need_rename_nodes = []
        for name, node in graph.node_map.items():
            for arg_name, outputs in node.outputs.items():
                for idx in range(len(outputs)):
                    if outputs[idx] == old_name:
                        node.outputs[arg_name][idx] = new_name
                        need_rename_nodes.append(node)
        for node in need_rename_nodes:
            graph.node_map[node.layer_name] = node
        return graph

    def mapping(self, onnx_graph):
        with paddle.static.program_guard(self.main_program,
                                         self.startup_program):
            res = self.forward()
        feed_var_names = [
            var.name for vars in self.inputs.values() for var in vars
        ]
        fetch_target_vars = [var for vars in res.values() for var in vars]
        inference_program = graph_helper.get_program(
            self.main_program, feed_var_names, fetch_target_vars)
        paddle_graph = PaddleGraph.build_from_program(
            inference_program, scope=fluid.global_scope())

        for arg_name, opts in res.items():
            for idx in range(len(opts)):
                new_name = self.node.output(arg_name, idx)
                old_name = opts[idx].name
                paddle_graph = self.rename_output_node(paddle_graph, old_name,
                                                       new_name)

        onnx_graph.build_parameters(paddle_graph.parameters)
        onnx_graph.build_op_nodes(paddle_graph.node_map)

    def insert_graph(self, node, graph):
        insert_index = list(self.node_map.keys()).index(node.layer_name)
        node_list = list(self.node_map.items())
        node_list.pop(insert_index)
        for idx, (name, node) in enumerate(graph.node_map.items()):
            if name in self.node_map:
                layer_name = self.generate_node_name(node.type)
                node.layer_name = layer_name
            node_list.insert(insert_index + idx, (node.layer_name, node))
        self.node_map = collections.OrderedDict(node_list)
