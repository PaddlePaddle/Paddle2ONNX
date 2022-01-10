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
        x = paddle.nn.functional.layer_norm(
            inputs,
            weight=None,
            bias=None,
            normalized_shape=self.config["normalized_shape"],
            epsilon=self.config["epsilon"])
        return x


class TestLayerNormConvert(OPConvertAutoScanTest):
    """
    api: paddle.nn.functional.layer_norm
    OPset version: 7, 9, 15
    """

    def sample_convert_config(self, draw):
        input_shape = draw(
            st.lists(
                st.integers(
                    min_value=2, max_value=20), min_size=2, max_size=4))

        input_spec = [-1] * len(input_shape)

        axis = draw(st.integers(min_value=1, max_value=len(input_shape) - 1))

        axis_type = draw(st.sampled_from(["int", "list"]))
        if axis_type == "int":
            normalized_shape = input_shape[-1]
        else:
            normalized_shape = input_shape[axis:]

        dtype = draw(st.sampled_from(["float32", "float64"]))
        epsilon = draw(st.floats(min_value=1e-12, max_value=1e-5))

        config = {
            "op_names": ["layer_norm"],
            "test_data_shapes": [input_shape],
            "test_data_types": [[dtype]],
            "opset_version": [7, 9, 15],
            "input_spec_shape": [],
            "epsilon": epsilon,
            "normalized_shape": normalized_shape
        }

        models = Net(config)

        return (config, models)

    def test(self):
        self.run_and_statis(max_examples=30)


if __name__ == "__main__":
    unittest.main()
