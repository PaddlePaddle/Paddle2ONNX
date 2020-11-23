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

from paddle2onnx.graph import Graph


class IfElsePass(object):
    logic_ops = ['equal', 'logical_not', 'greater_than', 'logic', 'less_than']

    def __init__(self):
        self.pattern_graph = self.build_pattern()

    def build_pattern(self):
        graph = Graph(0)
        logic = graph.make_node('logic_ops', layer_name='logic_ops', inputs=[])
        logic_not = graph.make_node('logical_not', inputs=[logic])
        cond_block = graph.make_node('conditional_block', inputs=[logic])
        cond_block_not = graph.make_node(
            'conditional_block', inputs=[logic_not])
        cast = graph.make_node('cast', inputs=[logic])
        graph.make_node(
            'select_input',
            inputs=[cond_block, cond_block_not, cast],
            outputs=[])
        graph.get_edge_map()

        return graph

    def match_node(self, node, pattern_node, graph):
        pattern_output_nodes = self.pattern_graph.get_output_nodes(pattern_node)
        if len(pattern_output_nodes) == 0:
            return True
        output_nodes = graph.get_output_nodes(node)
        for pattern_node in pattern_output_nodes:
            op_type = pattern_node.type
            matched = False
            for i in range(len(output_nodes)):
                if op_type == output_nodes[i].type:
                    if not self.match_node(output_nodes[i], pattern_node,
                                           graph):
                        return False
                    #output_nodes = output_nodes[:i] + output_nodes[i + 1:]
                    matched = True
            if not matched:
                return False
        return True

    def match_graph(self, graph):
        sub_node = self.pattern_graph.node_map['logic_ops']

        for name, node in graph.node_map.items():
            if node.type in self.logic_ops:
                #DFS traverse graph to check node match or not match 
                print('*********')
                self.match_node(node, sub_node, graph)
        for sub_graph in graph.sub_graphs:
            self.match_graph(sub_graph)
        return

    def passing(self, graph):
        matched = self.match_graph(graph)
        if matched:
            pass
        return matched
