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

    def __init__(self, config=None):
        super(Net, self).__init__(config)
        output_size = self.config['output_size']
        data_format = self.config['data_format']

        self.max_pool = paddle.nn.AdaptiveAvgPool3D(
            output_size, data_format=data_format)

    def forward(self, inputs):
        """
        forward
        """
        x = self.max_pool(inputs)
        return x


class TestGroupNormConvert(OPConvertAutoScanTest):
    """
    api: paddle.fluid.layers.nn.group_norm
    OPset version: 7, 9, 15
    """

    def sample_convert_config(self, draw):
        input_shape = draw(
            st.lists(
                st.integers(
                    min_value=4, max_value=10), min_size=4, max_size=4))

        input_shape = [3, 1, 10, 10, 10]
        dtype = draw(st.sampled_from(["float32"]))
        data_format = draw(st.sampled_from(["NCDHW"]))

        output_size = 3

        config = {
            "op_names": ["pool"],
            "test_data_shapes": [input_shape],
            "test_data_types": [[dtype]],
            "opset_version": [7, 9, 15],
            "input_spec_shape": [],
            "output_size": output_size,
            "data_format": data_format,
        }

        models = Net(config)

        return (config, models)

    def test(self):
        self.run_and_statis(max_examples=30)


if __name__ == "__main__":
    unittest.main()
