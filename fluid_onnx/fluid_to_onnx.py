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
# -*- coding:utf-8 -*-
import os
import sys
import argparse

from fluid.utils import op_io_info, init_name_prefix
from onnx import helper, checker
import paddle.fluid as fluid

import fluid_onnx.ops as ops
from fluid_onnx.variables import paddle_variable_to_onnx_tensor, paddle_onnx_weight
from debug.model_check import debug_model, Tracker


def parse_args():
    # Read arguments: path to model.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fluid_model", required=True, help="Input PaddlePaddle Fluid model.")
    parser.add_argument(
        "--onnx_model", required=True, help="The path to save ONNX model.")
    parser.add_argument(
        "--name_prefix", type=str, default="", help="The prefix of Var name.")
    parser.add_argument(
        "--fluid_model_name",
        type=str,
        default="",
        help="The fluid model name.")
    parser.add_argument(
        "--fluid_params_name",
        type=str,
        default="",
        help="The fluid params name.")
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Use the debug mode to validate the onnx model.")
    parser.add_argument(
        "--return_variable",
        action="store_true",
        default=True,
        help="If output is LoDTensor, the model outputs need to be variable.")
    parser.add_argument(
        "--check_task",
        type=str,
        default="image_detection_yolo",
        help="Use the different reader and backend to run the program, including image_classification, image_detection_ssd and image_detection_yolo"
    )
    parser.add_argument(
        "--image_path",
        type=str,
        default="",
        help="The image path to validate.")
    parser.add_argument(
        "--to_print_model",
        action='store_true',
        help="To print converted ONNX model.")
    args = parser.parse_args()
    return args


def print_arguments(args):
    print('-----------  Configuration Arguments -----------')
    for arg, value in sorted(vars(args).items()):
        print('%s: %s' % (arg, value))
    print('------------------------------------------------')


def convert(args):
    # Read the model files.
    place = fluid.CPUPlace()
    exe = fluid.Executor(place)

    inference_scope = fluid.core.Scope()
    with fluid.scope_guard(inference_scope):

        # Load inference program and other target attributes
        if len(args.fluid_model_name) != 0 and len(args.fluid_params_name) != 0:
            [inference_program, feed_target_names,
             fetch_targets] = fluid.io.load_inference_model(
                 args.fluid_model, exe, args.fluid_model_name,
                 args.fluid_params_name)
        else:
            [inference_program, feed_target_names, fetch_targets
             ] = fluid.io.load_inference_model(args.fluid_model, exe)

        fetch_targets_names = [data.name for data in fetch_targets]

        feed_fetch_list = ["fetch", "feed"]
        if args.name_prefix:
            feed_fetch_list = [
                args.name_prefix + name for name in feed_fetch_list
            ]

        # Load parameters
        weights, weights_value_info = [], []
        global_block = inference_program.global_block()
        for var_name in global_block.vars:
            var = global_block.var(var_name)
            if var_name not in feed_fetch_list\
                and var.persistable:
                weight, val_info = paddle_onnx_weight(
                    var=var, scope=inference_scope)
                weights.append(weight)
                weights_value_info.append(val_info)

        # Create inputs
        inputs = [
            paddle_variable_to_onnx_tensor(v, global_block)
            for v in feed_target_names
        ]

        print("load the model parameter done.")
        # Create nodes using blocks in inference_program
        init_name_prefix(args.name_prefix)
        onnx_nodes = []
        op_check_list = []
        op_trackers = []
        nms_first_index = -1
        nms_outputs = []
        for block in inference_program.blocks:
            for op in block.ops:
                if op.type in ops.node_maker:
                    # TODO(kuke): deal with the corner case that vars in 
                    #     different blocks have the same name
                    node_proto = ops.node_maker[str(op.type)](operator=op,
                                                              block=block)
                    op_outputs = []
                    last_node = None
                    if isinstance(node_proto, tuple):
                        onnx_nodes.extend(list(node_proto))
                        last_node = list(node_proto)
                    else:
                        onnx_nodes.append(node_proto)
                        last_node = [node_proto]
                    tracker = Tracker(str(op.type), last_node)
                    op_trackers.append(tracker)
                    op_check_list.append(str(op.type))
                    if op.type == "multiclass_nms" and nms_first_index < 0:
                        nms_first_index = 0
                    if nms_first_index >= 0:
                        _, _, output_op = op_io_info(op)
                        for output in output_op:
                            nms_outputs.extend(output_op[output])
                else:
                    if op.type not in ['feed', 'fetch']:
                        op_check_list.append(op.type)
                        #raise NotImplementedError("OP[%s] is not supported in "
                        #                         "the converter!" % op.type)
        print('The operator sets to run test case.')
        print(set(op_check_list))
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
                # If in debug mode, need to save op list, add we will check op 
                if args.debug:
                    op_check_list = list(set(op_check_list))
                    check_outputs = []

                    for node_proto in onnx_nodes:
                        check_outputs.extend(node_proto.output)

                    print("The num of %d operators need to check, and %d op outputs need to check."\
                          %(len(op_check_list), len(check_outputs)))

                    debug_model(op_check_list, op_trackers, nms_outputs, args)

            except Exception as e:
                print(e)
                print(
                    "Convert Failed! Please use the debug message to find error."
                )
                sys.exit(-1)


def main():
    args = parse_args()
    print_arguments(args)
    convert(args)


if __name__ == "__main__":
    main()
