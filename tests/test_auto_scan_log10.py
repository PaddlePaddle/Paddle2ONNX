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

from auto_scan_test import OPConvertAutoScanTest
import numpy as np
import paddle.inference as paddle_infer
from functools import partial
from typing import Optional, List, Callable, Dict, Any, Set
import unittest
import paddle

import hypothesis
from hypothesis import given, settings, seed, example, assume, reproduce_failure
import hypothesis.strategies as st


class Net(paddle.nn.Layer):
    """
    simple Net
    """

    def __init__(self, config):
        super(Net, self).__init__()

    def forward(self, inputs):
        """
        forward
        """
        x = paddle.log10(inputs)
        return x


class TestLog10Convert(OPConvertAutoScanTest):
    """
    api: paddle.nn.Conv2d
    OPset version: 9
    """

    def sample_convert_config(self, draw):
        input_shape = draw(
            st.lists(
                st.integers(
                    min_value=20, max_value=100),
                min_size=1,
                max_size=4))

        dtype = draw(st.sampled_from(["float32", "float64"]))

        config = {"input_shape": input_shape, }

        self.model = Net(config)
        self.op_name = "log10"
        self.test_data_shape = [input_shape]
        self.test_data_type = [[dtype]]
        self.input_spec_shape = []

        return config

    def test(self):
        self.run_and_statis(max_examples=25, opset_version=[9])


if __name__ == "__main__":
    unittest.main()
