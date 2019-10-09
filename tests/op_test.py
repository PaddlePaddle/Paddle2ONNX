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

from caffe2.python.onnx.backend import Caffe2Backend
import sys
import os
import unittest
import numpy as np

from onnx import checker
from onnx.helper import make_node, make_graph, make_model
from onnx.checker import check_node
import paddle.fluid.core as core
from paddle.fluid import scope_guard
from paddle.fluid.backward import append_backward
from paddle.fluid.op import Operator
from paddle.fluid.executor import Executor
from paddle.fluid.framework import Program, OpProtoHolder

import fluid_onnx.ops as ops
from fluid_onnx.variables import paddle_variable_to_onnx_tensor, PADDLE_TO_ONNX_DTYPE
from onnx import helper
import pickle
import time


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
    """
    NOTE (varunarora): Some of the code snippets below have been inspired from
    op_test.py in /python/paddle/fluid/tests/unittests/ in the original
    Paddle repository (https://github.com/PaddlePaddle/Paddle/).

    When in doubt, keep in sync with it's counterparts.
    """

    def append_input_output(self, block, op_proto, np_list, persistable_list,
                            is_input):
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
            """Creates a Paddle var in the given block and C++ proto type.
            """

            # If the expected variable is not found is in the provided list
            # of variables, make an assertion. Else, determine the shape and
            # and set the LoD level before creating the Paddle variable.
            if name not in np_list:
                #assert var_proto.intermediate, "{} not found".format(name)
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
                dtype='float32',
                shape=shape,
                persistable=persistable,
                lod_level=lod_level,
                name=name)

        # Go through all the variables in the expected list for this operator.
        var_dict = {}
        for var_proto in proto_list:
            var_name = str(var_proto.name)

            # If these are inputs, and the expected input is not necessary
            # and not provided in the list of inputs, we move on to the next
            # expected input. If not, we make sure it the expected input is 
            # provided, or that it is unnecessary.
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
                var_dict[var_name] = create_var(block, var_name, np_list,
                                                var_proto)

        return var_dict

    def create_tensor(self, np_value, place):
        """Create a LoDTensor initialized by the numpy ndarray.

        Args: 
            np_value (ndarray|tuple): The numpy ndarry to initialize the tensor, 
                                      in tuple (value, LoD) when LoD is given.
            place (CPUPlace|CUDAPlace): The place for the tensor.
        Return:
            The created LoDTensor.
        """

        tensor = core.LoDTensor()
        if isinstance(np_value, tuple):
            tensor.set(np_value[0], place)
            tensor.set_lod(np_value[1])
        else:
            tensor.set(np_value, place)

        return tensor

    def feed_var(self, input_vars, place):
        """Returns a dictionary of variable names -> initialized tensors.

        It sets tensors' execution place set (CPU or GPU), and Level of
        Details (LoD) using this info from the numpy values.
        """

        feed_map = {}
        for var_name in input_vars:
            if isinstance(input_vars[var_name], list):
                for name, np_value in self.inputs[var_name]:
                    tensor = self.create_tensor(np_value, place)
                    feed_map[name] = tensor
            else:
                tensor = self.create_tensor(self.inputs[var_name], place)
                feed_map[var_name] = tensor

        return feed_map

    def run_executor_return_tensor(self, exe, program):
        outs = exe.run(program,
                       feed=self.feed_map,
                       fetch_list=self.fetch_list,
                       return_numpy=False)

        outs = [np.array(out) for out in outs]
        return outs

    def eval_fluid_op(self, no_check_set, return_numpy):
        """Run a Paddle program only with the op to test.

        Returns the output values after running.
        """

        op_proto = OpProtoHolder.instance().get_op_proto(self.op_type)

        # Create a new paddle scope and program.
        place = core.CPUPlace()
        exe = Executor(place)
        scope = core.Scope()

        with scope_guard(scope):
            program = Program()
            self.block = program.global_block()

            # A list of inputs and outputs used by the op
            # that need to persisted in the global block.
            persistable = self.persistable if hasattr(self,
                                                      "persistable") else []

            # Add input and output variables to the global block.
            inputs = self.append_input_output(self.block, op_proto, self.inputs,
                                              persistable, True)
            outputs = self.append_input_output(self.block, op_proto,
                                               self.outputs, persistable, False)

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

            # A list containing outputs that wouldn't be used as outputs 
            # of ONNX node
            ignored_outputs = self.ignored_outputs if hasattr(
                self, "ignored_outputs") else []

            # Construct a unique list of outputs to fetch.
            self.fetch_list = []
            for var_name, var in outputs.items():
                if var_name in no_check_set:
                    continue
                if var_name in self.outputs and var_name not in ignored_outputs:
                    if isinstance(var, list):
                        for v in var:
                            self.fetch_list.append(v)
                    else:
                        self.fetch_list.append(var)

            self.feed_map = self.feed_var(inputs, place)

            if return_numpy:
                outs = exe.run(program,
                               feed=self.feed_map,
                               fetch_list=self.fetch_list,
                               return_numpy=True)
            else:
                start_time = time.time()
                outs = self.run_executor_return_tensor(exe, program)
                end_time = time.time()
        return outs

    def eval_onnx_node(self, no_check_set):
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
            fetch_target.name for fetch_target in self.fetch_list \
            if fetch_target.name not in no_check_set
        ]
        outputs = [
            paddle_variable_to_onnx_tensor(v, self.block)
            for v in fetch_target_names
        ]
        # Construct the ONNX model using paddle-onnx.
        onnx_node = ops.node_maker[self.op_type](operator=self.op,
                                                 block=self.block)

        node_list = list(onnx_node) if isinstance(onnx_node,
                                                  tuple) else [onnx_node]

        for node in node_list:
            check_node(node)
        #onnx_graph = make_graph(node_list, self.op_type, inputs, vars)
        onnx_graph = make_graph(node_list, self.op_type, inputs, outputs)
        onnx_model = make_model(onnx_graph, producer_name='unittest')

        # Expand input dictionary if there are tensor arrays
        input_map = {}
        for v in self.inputs:
            if isinstance(self.inputs[v], list):
                input_map.update(self.inputs[v])
            else:
                input_map[v] = self.inputs[v]

        # Run the Caffe2Backend with the ONNX model.
        rep = Caffe2Backend.prepare(onnx_model, device='CPU')
        in_vals = [input_map[input.name] for input in inputs]
        outs = rep.run(in_vals)
        return outs

    def check_onnx_with_onnxruntime(self,
                                    no_check_set=[],
                                    is_slice=False,
                                    is_nearest_interp=False,
                                    is_bilinear=False):
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
            fetch_target.name for fetch_target in self.fetch_list \
            if fetch_target.name not in no_check_set
        ]
        outputs = [
            paddle_variable_to_onnx_tensor(v, self.block)
            for v in fetch_target_names
        ]

        # Construct the ONNX model using paddle-onnx.
        onnx_node = ops.node_maker[self.op_type](operator=self.op,
                                                 block=self.block)
        #onnx_node, vars = ops.node_maker[self.op_type](operator=self.op,
        #                                         block=self.block)
        node_list = list(onnx_node) if isinstance(onnx_node,
                                                  tuple) else [onnx_node]
        for node in node_list:
            check_node(node)
        onnx_graph = make_graph(node_list, self.op_type, inputs, outputs)
        onnx_model = make_model(onnx_graph, producer_name='unittest')
        if is_slice:
            with open("tests/onnx_op/slice_test.onnx", 'wb') as f:
                f.write(onnx_model.SerializeToString())
        elif is_nearest_interp:
            with open("tests/onnx_op/nearest_interp_test.onnx", 'wb') as f:
                f.write(onnx_model.SerializeToString())
        elif is_bilinear:
            with open("tests/onnx_op/bilinear_interp_test.onnx", 'wb') as f:
                f.write(onnx_model.SerializeToString())
        else:
            with open("tests/nms_test.onnx", 'wb') as f:
                f.write(onnx_model.SerializeToString())
        checker.check_model(onnx_model)
        # Expand input dictionary if there are tensor arrays
        input_map = {}
        for v in self.inputs:
            if isinstance(self.inputs[v], list):
                input_map.update(self.inputs[v])
            else:
                input_map[v] = self.inputs[v]
        in_vals = [input_map[input.name] for input in inputs]
        with open("tests/inputs_test.pkl", "wb") as f:
            pickle.dump(input_map, f)
        ret = os.system("python tests/onnx_runtime.py %s %s %s" %
                        (is_slice, is_nearest_interp, is_bilinear))
        #if is_slice:
        #    ret = os.system("python tests/slice_onnx_runtime.py")
        #elif is_nearest_interp:
        #    ret = os.system("python tests/nearest_onnx_runtime.py")
        #else:
        #    ret = os.system("python tests/onnx_runtime.py")
        with open("tests/outputs_test.pkl", "rb") as f:
            outputs_val = pickle.load(f)
        f.close()
        return outputs_val

    def check_output(self,
                     no_check_set=[],
                     decimal=6,
                     return_numpy=True,
                     is_nms=False,
                     is_slice=False,
                     is_nearest_interp=False,
                     is_bilinear=False):
        """Compares the outputs from the Paddle program and the Caffe2
        backend using the ONNX model constructed by paddle-onnx.

        Compares accuracy at a precision of 5 decimal places by default.
        Because caffe2 do not support the nms op, we just use onnxruntime to test
        """

        fluid_result = self.eval_fluid_op(no_check_set, return_numpy)
        if is_slice or is_nearest_interp or is_bilinear:
            onnx_result = self.check_onnx_with_onnxruntime(
                is_slice=is_slice, is_nearest_interp=is_nearest_interp, is_bilinear=is_bilinear)

        elif is_nms:
            onnx_result = self.check_onnx_with_onnxruntime()
            onnx_result = onnx_result[0]
            onnx_result = np.squeeze(onnx_result, axis=0)
            fluid_result = fluid_result[0]
            fluid_result = fluid_result[(-fluid_result[:, 1]).argsort()]
        else:
            onnx_result = self.eval_onnx_node(no_check_set)
        for ref, hyp in zip(fluid_result, onnx_result):
            # Compare the values using numpy's almost_equal comparator.
            #print("ref:", ref)
            #print("hyp:", hyp)
            np.testing.assert_almost_equal(ref, hyp, decimal=decimal)

    def check_onnx_result(self, node_message, fetch_targets, return_numpy=True):
        fluid_result = self.eval_fluid_op([], return_numpy)
        onnx_result = self.eval_onnx_node_with_output_message(node_message,
                                                              fetch_targets)

    def check_intermedidate_result(self):
        fluid_result = self.eval_fluid_op([], True)
        onnx_result = self.check_onnx_with_onnxruntime(no_check_set=[])
