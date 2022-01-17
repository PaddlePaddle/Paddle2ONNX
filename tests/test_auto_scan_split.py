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

    def forward(self, inputs):
        """
        forward
        """
        axis = self.config['axis']
        if self.config['isAxisTensor']:
            axis = paddle.to_tensor(axis, dtype=self.config['axis_dtype'])
        x = paddle.split(
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
                    min_value=2, max_value=8), min_size=2, max_size=5))
        # float64 not supported
        dtype = draw(st.sampled_from(["float32", "int32", "int64"]))

        axis_dtype = draw(st.sampled_from(["int32", "int64"]))
        isAxisTensor = draw(st.booleans())
        axis = draw(
            st.integers(
                min_value=-len(input_shape), max_value=len(input_shape) - 1))

        axis_index = axis
        if axis < 0:
            axis_index = len(input_shape) + axis
        num_or_sections_dtype = draw(st.sampled_from(["int", "list"]))
        if num_or_sections_dtype == "list":
            x = [i + 1 for i in range(input_shape[axis_index])]
            data_list = self.generate_data(x)
            index = draw(st.integers(min_value=0, max_value=len(data_list) - 1))
            num_or_sections = list(data_list[index])
            random.shuffle(num_or_sections)
            if draw(st.booleans()):
                if len(num_or_sections) == 1:
                    num_or_sections[0] = -1
                else:
                    idx = draw(
                        st.integers(
                            min_value=0, max_value=len(num_or_sections) - 1))
                    num_or_sections[idx] = -1
        else:
            num_or_sections = draw(st.integers(min_value=2, max_value=5))
            scale = draw(st.integers(min_value=2, max_value=3))
            input_shape[axis_index] = num_or_sections * scale

        config = {
            "op_names": ["split"],
            "test_data_shapes": [input_shape],
            "test_data_types": [[dtype]],
            "opset_version": [7, 9, 15],
            "input_spec_shape": [],
            "axis": axis,
            "axis_dtype": axis_dtype,
            "isAxisTensor": isAxisTensor,
            "num_or_sections": num_or_sections
        }

        models = Net(config)

        return (config, models)

    def test(self):
        self.run_and_statis(max_examples=30)

    def generate_data(self, input):
        def max_values(n, x):
            max_value = 0
            iters = np.arange(0, x, 1)
            for i in iters:
                max_value = np.power((n - x + 2), i) * (n - x + 1) + max_value
            return max_value

        def min_values(n, x):
            min_value = 0
            iters = np.arange(0, x, 1)
            for i in iters:
                min_value = np.power((n - x + 2), i) * 1 + min_value
            return min_value

        def f(n, x):
            a = range(1, 30, 1)
            b = []
            while True:
                s = n // x
                y = n % x
                b = b + [y]
                if s == 0:
                    break
                n = s
            b.reverse()
            b = list(b)
            return b

        a = list(set(input))
        iters = np.arange(1, a.__len__() + 1, 1)
        contianer = list()
        discontianer = list()
        for i in iters:
            for j in (np.arange(
                    min_values(a.__len__(), i),
                    max_values(a.__len__(), i) + 1, 1)):
                b = f(j, a.__len__() - i + 2)
                if sum(b) == a.__len__() and (0 not in b):
                    t = sorted(b)
                    contianer.append(tuple(t))
                    discontianer = list(set(contianer))
        return discontianer


if __name__ == "__main__":
    unittest.main()
