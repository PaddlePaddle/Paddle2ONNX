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
from random import shuffle
import random


class Net(BaseNet):
    """
    simple Net
    """

    def forward(self, inputs, index, updates):
        """
        forward
        """
        x = paddle.scatter_nd_add(inputs, index, updates)
        return x


class TestScatterndaddConvert1(OPConvertAutoScanTest):
    """
    api: paddle.scatter_nd_add
    OPset version: 7, 9, 15
    """

    def sample_convert_config(self, draw):
        input_shape = draw(
            st.lists(
                st.integers(
                    min_value=30, max_value=40), min_size=1, max_size=4))

        index_shape = draw(
            st.lists(
                st.integers(
                    min_value=2, max_value=3), min_size=1, max_size=2))

        Q = draw(st.integers(min_value=1, max_value=len(input_shape)))
        index_shape = index_shape + [Q]
        update_shape = index_shape[:-1] + input_shape[index_shape[-1]:]

        dtype = draw(st.sampled_from(["float32"]))
        index_dtype = draw(st.sampled_from(["int64"]))

        def generator_index():
            prod = np.prod(index_shape)
            index_list = list(range(0, prod))
            shuffle(index_list)

            index_list = np.array(random.sample(index_list, prod))
            # index_list has duplicate, there has a diff
            index_list = index_list.reshape(index_shape)
            return index_list

        config = {
            "op_names": ["scatter_nd_add"],
            "test_data_shapes": [input_shape, generator_index, update_shape],
            "test_data_types": [[dtype], [index_dtype], [dtype]],
            "opset_version": [11, 12, 13, 14, 15],
            "input_spec_shape": [],
        }

        models = Net(config)

        return (config, models)

    def test(self):
        self.run_and_statis(max_examples=50)


if __name__ == "__main__":
    unittest.main()
