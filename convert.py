# Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.
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

import os
import argparse

from onnx import helper
import paddle.fluid as fluid

import ops
from variables import paddle_variable_to_onnx_tensor


def convert(dirname):
    # Read the model files.
    place = fluid.CPUPlace()
    exe = fluid.Executor(place)

    inference_scope = fluid.core.Scope()
    with fluid.scope_guard(inference_scope):
        [inference_program, feed_target_names,
            fetch_targets] = fluid.io.load_inference_model(dirname, exe)

        # Using blocks in programs, create nodes using:
        onnx_nodes = []
        all_inputs = []
        for block in inference_program.blocks:
            all_inputs += [paddle_variable_to_onnx_tensor(
                v, block) for v in block.vars if v not in ['feed', 'fetch']]

            for op in block.ops:
                if op.type in ops.PADDLE_TO_ONNX:
                    # TODO(varunarora): Attributes.
                    # TODO(varunarora): Use the modifier function to make the
                    # transformation.
                    node_proto = helper.make_node(
                        ops.PADDLE_TO_ONNX[op.type][0],
                        op.input_arg_names, op.output_arg_names)

                    onnx_nodes.append(node_proto)
                else:
                    # Not valid to skip any op, so after all edge cases have
                    # been accounted for, this exception raising to be
                    # re-enabled.
                    # raise NameError(op.type)
                    pass

        # Nodes, name of graph, inputs, outputs.
        if dirname[-1] == '/':
            dirname = dirname[:-1]
        graph = helper.make_graph(onnx_nodes, os.path.basename(
            dirname).split('.')[0], all_inputs, [])

        print graph

        # TODO(varunarora): Plug in parameters.


if __name__ == "__main__":
    # Read arguments: path to model.
    parser = argparse.ArgumentParser()
    parser.add_argument("--modeldir", required=True,
        help="Input PaddlePaddle model")
    args = parser.parse_args()
    convert(args.modeldir)
