# Copyright (c) 2021  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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

from auto_scan_test import OPConvertAutoScanTest, BaseNet
from hypothesis import reproduce_failure
import hypothesis.strategies as st
import numpy as np
import unittest
import paddle


class Net(BaseNet):
    """
    simple Net
    """

    def forward(self, inputs):
        """
        forward
        """
        axis = self.config['axis']
        if self.config['isAxisTensor']:
            axis = paddle.to_tensor(axis)
        x, out1, out2 = paddle.split(
            inputs, num_or_sections=self.config['num_or_sections'], axis=axis)
        return x


class TestSplitConvert(OPConvertAutoScanTest):
    """
    api: paddle.split
    OPset version: 7, 9, 15
    """

    def sample_convert_config(self, draw):
        input_shape = draw(
            st.lists(
                st.integers(
                    min_value=4, max_value=10), min_size=3, max_size=3))
        input_shape = [6, 6, 6]
        # float64 not supported
        dtype = draw(st.sampled_from(["float16", "float32", "int32", "int64"]))

        isAxisTensor = draw(st.booleans())
        axis = draw(
            st.integers(
                min_value=-len(input_shape), max_value=len(input_shape) - 1))
        axis_index = 0
        # if axis < 0:
        #     axis_index = len(input_shape) + axis
        # tt = input_shape[axis_index]
        num_or_sections_dtype = draw(st.sampled_from(["int", "list"]))
        if num_or_sections_dtype == "list":
            num_or_sections = [2, 2, 2]
        else:
            num_or_sections = 3

        axis = 2
        config = {
            "op_names": ["split"],
            "test_data_shapes": [input_shape],
            "test_data_types": [[dtype]],
            "opset_version": [7, 9, 15],
            "input_spec_shape": [],
            "axis": axis,
            "isAxisTensor": isAxisTensor,
            "num_or_sections": num_or_sections
        }

        models = Net(config)

        return (config, models)

    def test(self):
        self.run_and_statis(max_examples=30)


if __name__ == "__main__":
    unittest.main()
