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


def get_repeated_output(node):
    repeated_output = {}
    ipt_names = set()
    for arg_name, inputs in node.inputs.items():
        for ipt in inputs:
            ipt_names.add(ipt)

    opt_to_arg_name = {}
    for arg_name, outputs in node.outputs.items():
        for idx in range(len(outputs)):
            opt = outputs[idx]
            if opt in ipt_names:
                repeated_output[opt] = (arg_name, idx)

    return repeated_output


@PassManager('inplace_node_pass')
class InplaceNodePass(object):

    name_count = dict()

    @classmethod
    def generate_name(cls, name):
        if name in cls.name_count:
            cls.name_count[name] += 1
        else:
            cls.name_count[name] = 1
        new_name = name + '.' + str(cls.name_count[name])
        return new_name

    @classmethod
    def run_pass(cls, paddle_graph):
        output_to_nodes = {}
        node_map = list(paddle_graph.node_map.items())
        for idx in range(len(node_map)):
            name, node = node_map[idx]
            repeated_output = get_repeated_output(node)
            if len(repeated_output) == 0:
                continue
            else:
                for name, (arg_name, idx) in repeated_output.items():
                    new_name = cls.generate_name(name)
                    node.outputs[arg_name][idx] = new_name
                paddle_graph.update_node(node)

        return paddle_graph
