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
import random


class Net(BaseNet):
    """
    simple Net
    """

    def forward(self, input, k):
        """
        forward
        """
        # has a bug
        # k = self.config['k']
        # if self.config['isTensor']:
        #     k = paddle.to_tensor(k, dtype="int32")
        x = paddle.topk(
            input,
            k=k,
            axis=self.config['axis'],
            largest=self.config['largest'],
            sorted=self.config['sorted'])
        return x


class TestTopkv2Convert(OPConvertAutoScanTest):
    """
    api: paddle.topk
    OPset version: 11, 15
    """

    def sample_convert_config(self, draw):
        input_shape = draw(
            st.lists(
                st.integers(
                    min_value=2, max_value=5), min_size=1, max_size=5))

        axis = None
        if draw(st.booleans()):
            axis = draw(
                st.integers(
                    min_value=-len(input_shape), max_value=len(input_shape) -
                    1))

        dtype = draw(st.sampled_from(["float32", "float64", "int32", "int64"]))

        largest = draw(st.booleans())
        sorted = True  # has a diff when sorted is False

        def generator_data():
            t = 1
            for i in range(len(input_shape)):
                t = t * input_shape[i]
            input_data = np.array(random.sample(range(-5000, 5000), t))
            input_data = input_data.reshape(input_shape)
            return input_data

        k = random.randint(1, min(input_shape))

        # isTensor = draw(st.booleans())
        def generator_k():
            input_data = np.array([k])
            return input_data

        config = {
            "op_names": ["top_k_v2"],
            "test_data_shapes": [generator_data, generator_k],
            "test_data_types": [[dtype], ["int32"]],
            "opset_version": [11, 15],
            "input_spec_shape": [],
            "axis": axis,
            "largest": largest,
            'sorted': sorted,
            # 'isTensor': isTensor,
            'k': k,
        }

        models = Net(config)

        return (config, models)

    def test(self):
        self.run_and_statis(max_examples=30)


class Net1(BaseNet):
    """
    simple Net
    """

    def forward(self, input):
        """
        forward
        """
        k = self.config['k']
        x = paddle.topk(
            input,
            k=k,
            axis=self.config['axis'],
            largest=self.config['largest'],
            sorted=self.config['sorted'])
        return x


class TestTopkv2Convert1(OPConvertAutoScanTest):
    """
    api: paddle.topk
    OPset version: 11, 15
    """

    def sample_convert_config(self, draw):
        input_shape = draw(
            st.lists(
                st.integers(
                    min_value=2, max_value=5), min_size=1, max_size=5))

        axis = None
        if draw(st.booleans()):
            axis = draw(
                st.integers(
                    min_value=-len(input_shape), max_value=len(input_shape) -
                    1))

        dtype = draw(st.sampled_from(["float32", "float64", "int32", "int64"]))

        largest = draw(st.booleans())
        # paddle sort is unstable
        sorted = True  # has a diff when sorted is False

        def generator_data():
            t = 1
            for i in range(len(input_shape)):
                t = t * input_shape[i]
            input_data = np.array(random.sample(range(-5000, 5000), t))
            input_data = input_data.reshape(input_shape)
            return input_data

        k = random.randint(1, min(input_shape))
        config = {
            "op_names": ["top_k_v2"],
            "test_data_shapes": [generator_data],
            "test_data_types": [[dtype]],
            "opset_version": [11, 15],
            "input_spec_shape": [],
            "axis": axis,
            "largest": largest,
            'sorted': sorted,
            'k': k,
        }

        models = Net1(config)

        return (config, models)

    def test(self):
        self.run_and_statis(max_examples=30)


if __name__ == "__main__":
    unittest.main()