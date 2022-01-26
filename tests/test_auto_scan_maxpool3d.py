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
        data_format = self.config['data_format']
        x = paddle.nn.functional.max_pool3d(
            inputs,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            return_mask=return_mask,
            ceil_mode=ceil_mode,
            data_format=data_format)
        return x


class TestMaxpool3dConvert(OPConvertAutoScanTest):
    """
    api: paddle.nn.functional.max_pool3d
    OPset version: 7, 9, 15
    """

    def sample_convert_config(self, draw):
        input_shape = draw(
            st.lists(
                st.integers(
                    min_value=1, max_value=10), min_size=5, max_size=5))

        # input_shape = [3, 1, 10, 10, 10]
        dtype = draw(st.sampled_from(["float32"]))
        data_format = draw(st.sampled_from(["NCDHW"]))

        kernel_size = 2
        stride = None
        padding = 0
        ceil_mode = False
        return_mask = False
        data_format = 'NCDHW'

        config = {
            "op_names": ["pool3d"],
            "test_data_shapes": [input_shape],
            "test_data_types": [[dtype]],
            "opset_version": [7, 9, 15],
            "input_spec_shape": [],
            "kernel_size": kernel_size,
            "stride": stride,
            "padding": padding,
            "ceil_mode": ceil_mode,
            "return_mask": return_mask,
            "data_format": data_format,
        }

        models = Net(config)

        return (config, models)

    def test(self):
        self.run_and_statis(max_examples=30)


if __name__ == "__main__":
    unittest.main()
