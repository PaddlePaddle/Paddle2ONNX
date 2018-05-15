# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

from fluid.utils import op_io_info
from onnx import helper, checker
import paddle.fluid as fluid

import fluid_onnx.ops as ops
from fluid_onnx.variables import paddle_variable_to_onnx_tensor, paddle_onnx_weight


def parse_args():
    # Read arguments: path to model.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fluid_model", required=True, help="Input PaddlePaddle Fluid model.")
    parser.add_argument(
        "--onnx_model", required=True, help="The path to save ONNX model.")
    parser.add_argument(
        "--to_print_model",
        action='store_true',
        help="To print converted ONNX model.")
    args = parser.parse_args()
    return args


def print_arguments(args):
    print('-----------  Configuration Arguments -----------')
    for arg, value in sorted(vars(args).iteritems()):
        print('%s: %s' % (arg, value))
    print('------------------------------------------------')


def convert(args):
    # Read the model files.
    place = fluid.CPUPlace()
    exe = fluid.Executor(place)

    inference_scope = fluid.core.Scope()
    with fluid.scope_guard(inference_scope):

        # Load inference program and other target attributes
        [inference_program, feed_target_names,
         fetch_targets] = fluid.io.load_inference_model(args.fluid_model, exe)

        # Load parameters
        weights, weights_value_info = [], []
        global_block = inference_program.global_block()
        for var_name in global_block.vars:
            var = global_block.var(var_name)
            if var_name not in ['feed', 'fetch'] and var.persistable:
                weight, val_info = paddle_onnx_weight(
                    var=var, scope=inference_scope)
                weights.append(weight)
                weights_value_info.append(val_info)

        # Create inputs
        inputs = [
            paddle_variable_to_onnx_tensor(v, global_block)
            for v in feed_target_names
        ]

        # Create nodes using blocks in inference_program
        onnx_nodes = []
        for block in inference_program.blocks:
            for op in block.ops:
                if op.type in ops.node_maker:
                    # TODO(kuke): deal with the corner case that vars in 
                    #     different blocks have the same name
                    node_proto = ops.node_maker[op.type](operator=op,
                                                         block=block)

                    if isinstance(node_proto, tuple):
                        onnx_nodes.extend(list(node_proto))
                    else:
                        onnx_nodes.append(node_proto)
                else:
                    if op.type not in ['feed', 'fetch']:
                        raise NotImplementedError("OP[%s] is not supported in "
                                                  "the converter!" % op.type)

        # Create outputs
        fetch_target_names = [
            fetch_target.name for fetch_target in fetch_targets
        ]
        # Get the new names for outputs if they've been renamed in nodes' making
        renamed_outputs = op_io_info.get_all_renamed_outputs()
        fetch_target_names = [
            name if name not in renamed_outputs else renamed_outputs[name]
            for name in fetch_target_names
        ]
        outputs = [
            paddle_variable_to_onnx_tensor(v, global_block)
            for v in fetch_target_names
        ]

        # Make graph
        model_name = os.path.basename(args.fluid_model.strip('/')).split('.')[0]
        onnx_graph = helper.make_graph(
            nodes=onnx_nodes,
            name=model_name,
            initializer=weights,
            inputs=inputs + weights_value_info,
            outputs=outputs)

        # Make model
        onnx_model = helper.make_model(onnx_graph, producer_name='PaddlePaddle')

        # Model check
        checker.check_model(onnx_model)

        # Print model
        if args.to_print_model:
            print("The converted model is:\n{}".format(onnx_model))

        # Save converted model
        if args.onnx_model is not None:
            try:
                with open(args.onnx_model, 'wb') as f:
                    f.write(onnx_model.SerializeToString())
                print("Saved converted model to path: %s" % args.onnx_model)
            except (IOError), e:
                print("Invalid ONNX model saving path: %s" % args.onnx_model)


if __name__ == "__main__":
    args = parse_args()
    print_arguments(args)
    convert(args)
