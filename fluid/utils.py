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

from compiler.ast import flatten


class OpIOsInfo():
    """Return inputs/outputs information for an operator, and resolve potential 
       name conflicts in ONNX graph.
    """

    def __init__(self):
        self._all_renamed_outputs = {}
        self._renamed_cnt = 0

    def _get_new_name(self, arg):
        """Get the new name for an argument.
        """

        self._renamed_cnt += 1
        return arg + '@dup_' + str(self._renamed_cnt)

    def _rename_input_args(self):
        """Rename input arguments if their previous output arugments have been 
           renamed.
        """

        for in_name in self.inputs:
            if self.inputs[in_name][0] in self._all_renamed_outputs:
                self.inputs[in_name][0] = self._all_renamed_outputs[self.inputs[
                    in_name][0]]

    def _rename_output_args(self):
        """Rename output arguments if they have same names with the input 
           arguments.
        """

        input_args = flatten(self.inputs.values())
        for out_name in self.outputs:
            if self.outputs[out_name][0] in input_args:
                new_name = self._get_new_name(self.outputs[out_name][0])
                self._all_renamed_outputs[self.outputs[out_name][0]] = new_name
                self.outputs[out_name][0] = new_name

    def get_all_renamed_outputs(self):
        """Get all the renamed outputs in history.
        """

        return self._all_renamed_outputs

    def __call__(self, op):
        self.inputs = dict([(name, op.input(name)) for name in op.input_names])
        self.attrs = dict(
            [(name, op.attr(name))
             for name in op.attr_names]) if op.attr_names is not None else None
        self.outputs = dict(
            [(name, op.output(name)) for name in op.output_names])

        self._rename_input_args()
        self._rename_output_args()

        return self.inputs, self.attrs, self.outputs


# Instantiate the class to a callable object
op_io_info = OpIOsInfo()
