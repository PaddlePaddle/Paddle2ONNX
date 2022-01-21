#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

from paddle2onnx.passes import PassManager
from paddle2onnx.utils import logging


@PassManager('remove_isolated_node_pass')
class RemoveIsolatedNodePass(object):
    @classmethod
    def run_pass(cls, onnx_graph):
        node_map = list(onnx_graph.node_map.items())
        for idx in range(len(node_map)):
            name, node = node_map[idx]
            inputs = node.inputs
            outputs = node.outputs
            if len(inputs) > 0:
                continue

            keep = False
            for node in onnx_graph.output_nodes:
                if node.name in outputs:
                    keep = True
                    break
            if keep:
                continue

            keep = False
            for inner_idx in range(idx + 1, len(node_map)):
                inner_name, inner_node = node_map[inner_idx]
                inner_inputs = inner_node.inputs
                for out in outputs:
                    if out in inner_inputs:
                        keep = True
            if not keep:
                logging.info("Delete isolated node: {}".format(name))
                onnx_graph.remove_node_by_name(name)
        return onnx_graph
