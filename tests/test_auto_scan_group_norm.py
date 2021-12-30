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
        x = paddle.fluid.layers.nn.group_norm(
            inputs,
            groups=inputs.shape[1],
            epsilon=self.config['epsilon'],
            param_attr=None,
            bias_attr=None,
            act=None,
            data_layout='NCHW')
        return x


class TestGroupNormConvert(OPConvertAutoScanTest):
    """
    api: paddle.fluid.layers.nn.group_norm
    OPset version: 11, 15
    """

    def sample_convert_config(self, draw):
        input_shape = draw(
            st.lists(
                st.integers(
                    min_value=20, max_value=100),
                min_size=4,
                max_size=4))

        dtype = draw(st.sampled_from(["float32"]))

        epsilon = draw(st.floats(min_value=1e-12, max_value=1e-5))

        config = {
            "op_names": ["group_norm_0"],
            "test_data_shapes": [input_shape],
            "test_data_types": [[dtype]],
            "opset_version": [11, 15],
            "input_spec_shape": [],
            "epsilon": epsilon,
        }

        models = Net(config)

        return (config, models)

    def test(self):
        self.run_and_statis(max_examples=30)


if __name__ == "__main__":
    unittest.main()
