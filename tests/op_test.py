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

import sys
import unittest
import numpy as np

from onnx.helper import make_node, make_graph, make_model
from onnx.checker import check_node
import paddle.fluid.core as core
from paddle.fluid import scope_guard
from paddle.fluid.backward import append_backward
from paddle.fluid.op import Operator
from paddle.fluid.executor import Executor
from paddle.fluid.framework import Program, OpProtoHolder
from caffe2.python.onnx.backend import Caffe2Backend

from fluid_onnx.ops import node_maker
from fluid_onnx.variables import paddle_variable_to_onnx_tensor

"""
NOTE (varunarora): Some of the code snippets below have been inspired from
op_test.py in /python/paddle/fluid/tests/unittests/ in the original
Paddle repository (https://github.com/PaddlePaddle/Paddle/).

When in doubt, keep in sync with it's counterparts.
"""

def append_input_output(block, op_proto, np_list, persistable_list, is_input):
    """Returns a list of Paddle variables associated with a block.

    Args:
        block:
        op_proto: The matching C++ operator type.
        np_list: Dict of value names -> values.
        persistable_list: List of variables to be persisted.
        is_input: Boolean of if this is a set of inputs.

    Returns:
        A dict of variable names -> Paddle variable instances.
    """
    # A list of expected inputs and outputs, as desired by Paddle's
    # C++ runtime.
    proto_list = op_proto.inputs if is_input else op_proto.outputs

    def create_var(block, name, np_list, var_proto):
        """Creates a Paddle var in the given block and C++ proto type"""

        # If the expected variable is not found is in the provided list
        # of variables, make an assertion. Else, determine the shape and
        # and set the LoD level before creating the Paddle variable.
        if name not in np_list:
            assert var_proto.intermediate, "{} not found".format(name)
            shape = None
            lod_level = None
        else:
            np_value = np_list[name]
            if isinstance(np_value, tuple):
                shape = list(np_value[0].shape)
                lod_level = len(np_value[1])
            else:
                shape = list(np_value.shape)
                lod_level = 0

        persistable = True if name in persistable_list else False
        return block.create_var(
            dtype="float32",
            shape=shape,
            persistable=persistable,
            lod_level=lod_level,
            name=name)

    # Go through all the variables in the expected list for this operator.
    var_dict = {}
    for var_proto in proto_list:
        var_name = str(var_proto.name)

        # If these are inputs, and the expected input is not necessary
        # are is not provided in the list of inputs, we move on to the next
        # expected. input.
        # If not, we make sure it the expected input is provided, or that it
        # is unnecessary.
        if is_input:
            if (var_name not in np_list) and var_proto.dispensable:
                continue
            assert (var_name in np_list) or (var_proto.dispensable), \
                "Missing {} as input".format(var_name)

        # Set duplicable variables as lists of Paddle variables, and standard
        # ones as simple Paddle variables.
        if var_proto.duplicable:
            assert isinstance(np_list[var_name], list), \
                "Duplicable {} should be set as list".format(var_name)
            var_list = []
            for (name, np_value) in np_list[var_name]:
                var_list.append(
                    create_var(block, name, {name: np_value}, var_proto))
            var_dict[var_name] = var_list
        else:
            var_dict[var_name] = create_var(block, var_name, np_list, var_proto)

    return var_dict


class OpTest(unittest.TestCase):
    """Evaluates an op maker's validity.

    Using op-specific inputs and attributes, executes:
    (1) A Paddle program
    (2) A Caffe2 backend consuming Paddle ops converted to ONNX using custom
        op makers.

    It uses the outputs of both fo these executions to compare the values
    of their outputs. Success in these comparisons comes from almost equal
    values of this output data across both executions.

    Attributes:
        inputs (dict): Operator input name -> input value.
        outputs (dict): Operator output -> output value placeholders.

        Additionally, custom attributes to the op.
    """
    def feed_var(self, input_vars, place):
        """Returns a dictionary of variable names -> initialized tensors.

        It sets tensors' execution place set (CPU or GPU), and Level of
        Details (LoD) using this info from the numpy values.
        """
        feed_map = {}
        for var_name in input_vars:
            if isinstance(input_vars[var_name], list):
                for name, np_value in self.inputs[var_name]:
                    tensor = core.LoDTensor()
                    if isinstance(np_value, tuple):
                        tensor.set(np_value[0], place)
                        tensor.set_lod(np_value[1])
                    else:
                        tensor.set(np_value, place)
                    feed_map[name] = tensor
            else:
                tensor = core.LoDTensor()
                if isinstance(self.inputs[var_name], tuple):
                    tensor.set(self.inputs[var_name][0], place)
                    tensor.set_lod(self.inputs[var_name][1])
                else:
                    tensor.set(self.inputs[var_name], place)
                feed_map[var_name] = tensor

        return feed_map

    def eval_fluid_op(self):
        """Run a Paddle program only with the op to test.

        Returns the output values after running.
        """
        op_proto = OpProtoHolder.instance().get_op_proto(self.op_type)

        # Create a new paddle scope and program.
        place = core.CPUPlace()
        exe = Executor(place)
        self.scope = core.Scope()

        with scope_guard(self.scope):
            program = Program()
            self.block = program.global_block()

            # A list of inputs and outputs used by the op
            # that need to persisted in the global block.
            persistable = self.persistable if hasattr(self,
                                                      "persistable") else []

            # Add input and output variables to the global block.
            inputs = append_input_output(self.block, op_proto, self.inputs,
                                         persistable, True)
            outputs = append_input_output(self.block, op_proto, self.outputs,
                                          persistable, False)

            # Append the op.
            self.op = self.block.append_op(
                type=self.op_type,
                inputs=inputs,
                outputs=outputs,
                attrs=self.attrs if hasattr(self, "attrs") else dict())

            # Infer the var type and share of the op based on the block's
            # inputs and outputs.
            self.op.desc.infer_var_type(self.block.desc)
            self.op.desc.infer_shape(self.block.desc)

            # Construct a unique list of outputs to fetch.
            self.fetch_list = []
            for var_name, var in outputs.iteritems():
                if var_name in self.outputs:
                    if isinstance(var, list):
                        for v in var:
                            self.fetch_list.append(v)
                    else:
                        self.fetch_list.append(var)

            self.feed_map = self.feed_var(inputs, place)

            outs = exe.run(program,
                           feed=self.feed_map,
                           scope=self.scope,
                           fetch_list=self.fetch_list,
                           return_numpy=True)
        return outs

    def eval_onnx_node(self):
        """Run a Caffe2 program using their ONNX backend.

        Prior to running the backend, use the Paddle scope to construct
        ONNX ops and prepare the inputs and output values based on ONNX
        compatibility.
        """
        # Convert inputs and outputs to ONNX tensors.
        # Use the Paddle fetch_list to prepare the outputs.
        inputs = [
            paddle_variable_to_onnx_tensor(v, self.block) for v in self.feed_map
        ]

        fetch_target_names = [
            fetch_target.name for fetch_target in self.fetch_list
        ]
        outputs = [
            paddle_variable_to_onnx_tensor(v, self.block)
            for v in fetch_target_names
        ]

        # Construct the ONNX model using paddle-onnx.
        onnx_node = node_maker[self.op_type](operator=self.op, scope=self.scope)
        node_list = list(onnx_node) if isinstance(onnx_node,
                                                  tuple) else [onnx_node]
        for node in node_list:
            check_node(node)

        onnx_graph = make_graph(node_list, self.op_type, inputs, outputs)
        onnx_model = make_model(onnx_graph, producer_name='unittest')

        # Run the Caffe2Backend with the ONNX model.
        rep = Caffe2Backend.prepare(onnx_model, device='CPU')
        in_vals = [self.inputs[input.name] for input in inputs]
        outs = rep.run(in_vals)

        return outs

    def check_output(self, decimal=5):
        """Compares the outputs from the Paddle program and the Caffe2
        backend using the ONNX model constructed by paddle-onnx.

        Compares accuracy at a precision of 5 decimal places by default.
        """
        fluid_result = self.eval_fluid_op()
        onnx_result = self.eval_onnx_node()

        for ref, hyp in zip(fluid_result, onnx_result):
            # Compare the values using numpy's almost_equal comparator.
            np.testing.assert_almost_equal(ref, hyp, decimal=decimal)
