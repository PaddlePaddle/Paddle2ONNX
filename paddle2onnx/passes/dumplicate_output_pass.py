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


def get_repeated_output(outputs_1, outputs_2):
    repeated_output = {}
    for idx in range(len(outputs_2)):
        opt = outputs_2[idx]
        if opt in outputs_1:
            repeated_output[opt] = idx
    return repeated_output


@PassManager('dumplicate_output_pass')
class DumplicateOutputPass(object):

    name_count = dict()

    @classmethod
    def generate_new_name(cls, name):
        if name in cls.name_count:
            cls.name_count[name] += 1
        else:
            cls.name_count[name] = 1
        new_name = name + '.' + str(cls.name_count[name])
        return new_name

    @classmethod
    def run_pass(cls, onnx_graph):
        node_map = list(onnx_graph.node_map.items())
        name_mapping = {}
        for index in range(len(node_map)):
            name_mapping.clear()
            name, node = node_map[index]
            inputs = node.inputs
            outputs = node.outputs

            for idx in range(index + 1, len(node_map)):
                inner_name, inner_node = node_map[idx]
                inner_inputs = inner_node.inputs
                inner_outputs = inner_node.outputs

                changed = False
                if len(name_mapping) > 0:
                    for i_dex in range(len(inner_inputs)):
                        ipt = inner_inputs[i_dex]
                        if ipt in name_mapping:
                            changed = True
                            inner_inputs[i_dex] = name_mapping[ipt]
                    for o_dex in range(len(inner_outputs)):
                        opt = inner_outputs[o_dex]
                        if opt in name_mapping:
                            changed = True
                            inner_outputs[o_dex] = name_mapping[opt]

                repeated_output = get_repeated_output(outputs, inner_outputs)
                if len(repeated_output) == 0:
                    continue
                if len(repeated_output) != 0:
                    for opt, o_dex in repeated_output.items():
                        name_mapping[opt] = cls.generate_new_name(opt)
                        inner_outputs[o_dex] = name_mapping[opt]
                        changed = True
                if changed:
                    node.set_inputs(inputs)
                    node.set_outputs(outputs)
                    onnx_graph.update_node(node)

        return onnx_graph
