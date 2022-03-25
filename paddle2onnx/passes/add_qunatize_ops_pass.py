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

from paddle2onnx.passes import PassManager


def get_repeated_output(inputs, outputs):
    repeated_output = {}
    for idx in range(len(outputs)):
        opt = outputs[idx]
        if opt in inputs:
            repeated_output[opt] = idx
    return repeated_output


@PassManager('add_quantize_ops_pass')
class AddQuantizeOpsPass(object):
    @classmethod
    def run_pass(cls, onnx_graph):
        if not onnx_graph.changed_dict:
            return
        sort_name_dict = dict()
        for input_name, vals in onnx_graph.changed_dict.items():
            ori_input_name = vals['name']
            total = vals['total']
            all_q_dq = vals['qdq']
            num = 0
            for layer_name, node in onnx_graph.node_map.items():
                inputs = node.inputs
                if input_name not in inputs:
                    continue
                if node.type in ['QuantizeLinear']:
                    continue
                if node.type.count("Pool"):
                    continue
                for i in range(len(inputs)):
                    if input_name == inputs[i]:
                        change_input_name = ori_input_name + ".paddleadd" + str(
                            num)
                        inputs[i] = change_input_name
                        node = onnx_graph.update_node(node, inputs=inputs)
                        sort_name_dict[node.layer_name] = all_q_dq[num]
                        num = num + 1

        temp_node_map = dict()
        for layer_name, node in onnx_graph.node_map.items():
            if layer_name in sort_name_dict.keys():
                qdq_node = sort_name_dict[layer_name]
                for fake_node_name in qdq_node:
                    fake_node = onnx_graph.node_map[fake_node_name]
                    temp_node_map[fake_node_name] = fake_node
                temp_node_map[layer_name] = node
            else:
                for fake_values in sort_name_dict.values():
                    if layer_name in fake_values:
                        continue
                    else:
                        temp_node_map[layer_name] = node
        onnx_graph.node_map = temp_node_map

        return onnx_graph
