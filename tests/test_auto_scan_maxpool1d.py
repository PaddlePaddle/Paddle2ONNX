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
        kernel_size = self.config['kernel_size']
        stride = self.config['stride']
        padding = self.config['padding']
        ceil_mode = self.config['ceil_mode']
        return_mask = self.config['return_mask']

        x = paddle.nn.functional.max_pool1d(
            inputs,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            return_mask=return_mask,
            ceil_mode=ceil_mode)
        return x


class TestMaxpool1dConvert(OPConvertAutoScanTest):
    """
    api: paddle.nn.functional.max_pool1d
    OPset version: 7, 9, 15
    """

    def sample_convert_config(self, draw):
        input_shape = draw(
            st.lists(
                st.integers(
                    min_value=10, max_value=20), min_size=3, max_size=3))

        # input_shape = [3, 1, 10]
        dtype = draw(st.sampled_from(["float32"]))

        return_mask = False
        ceil_mode = draw(st.booleans())

        kernel_type = draw(st.sampled_from(["int", "list"]))
        if kernel_type == "int":
            kernel_size = draw(st.integers(min_value=7, max_value=10))
        elif kernel_type == "list":
            kernel_size = draw(
                st.lists(
                    st.integers(
                        min_value=7, max_value=10),
                    min_size=1,
                    max_size=1))

        stride_type = draw(st.sampled_from(["None", "int", "list"]))
        if stride_type == "int":
            stride = draw(st.integers(min_value=1, max_value=5))
        elif stride_type == "list":
            stride = draw(
                st.lists(
                    st.integers(
                        min_value=1, max_value=5),
                    min_size=1,
                    max_size=1))
        else:
            stride = None

        padding = 0

        opset_version = [[7, 9, 15]]
        if ceil_mode:
            opset_version = [10, 15]

        if padding == "VALID":
            ceil_mode = False

        config = {
            "op_names": ["pool2d"],
            "test_data_shapes": [input_shape],
            "test_data_types": [[dtype]],
            "opset_version": opset_version,
            "input_spec_shape": [],
            "kernel_size": kernel_size,
            "stride": stride,
            "padding": padding,
            "ceil_mode": ceil_mode,
            "return_mask": return_mask,
        }

        models = Net(config)

        return (config, models)

    def test(self):
        self.run_and_statis(max_examples=30)


if __name__ == "__main__":
    unittest.main()
