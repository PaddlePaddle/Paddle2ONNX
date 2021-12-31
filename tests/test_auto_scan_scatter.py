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
from onnxbase import randtool


class Net(BaseNet):
    """
    simple Net
    """

    def forward(self, inputs, _index, _updates):
        """
        forward
        """
        x = paddle.scatter(inputs, _index, _updates, overwrite=True)
        return x


class TestScatterConvert(OPConvertAutoScanTest):
    """
    api: paddle.scatter
    OPset version: 11, 12
    """

    def sample_convert_config(self, draw):
        input_shape = draw(
            st.lists(
                st.integers(
                    min_value=4, max_value=10), min_size=2, max_size=2))
        input_shape = [3, 2]
        dtype = draw(st.sampled_from(["float32"]))
        update_shape = [3]

        def generator_data():
            input_data = randtool("int", 1, 2, update_shape)
            return input_data

        config = {
            "op_names": ["scatter"],
            "test_data_shapes": [input_shape, generator_data, input_shape],
            "test_data_types": [[dtype], ['int64'], [dtype]],
            "opset_version": [11, 12],
            "input_spec_shape": [],
        }

        models = Net(config)

        return (config, models)

    def test(self):
        self.run_and_statis(max_examples=30)


if __name__ == "__main__":
    unittest.main()