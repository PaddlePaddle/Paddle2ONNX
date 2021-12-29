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

    def forward(self, input):
        """
        forward
        """
        x = paddle.topk(
            input,
            k=self.config['k'],
            axis=self.config['axis'],
            largest=self.config['largest'],
            sorted=self.config['sorted'])
        return x[0]


class TestTopkv2Convert(OPConvertAutoScanTest):
    """
    api: paddle.topk
    OPset version: 7, 9, 15
    """

    def sample_convert_config(self, draw):
        input_shape = draw(
            st.lists(
                st.integers(
                    min_value=20, max_value=100),
                min_size=1,
                max_size=4))

        axis = None
        if draw(st.booleans()):
            axis = draw(
                st.integers(
                    min_value=-len(input_shape), max_value=len(input_shape) -
                    1))

        dtype = draw(st.sampled_from(["float32", "float64", "int32", "int64"]))

        largest = draw(st.booleans())
        sortedVal = draw(st.booleans())
        if draw(st.booleans()):
            k = 2
        else:
            k = paddle.to_tensor(1)

        config = {
            "op_names": ["top_k_v2"],
            "test_data_shapes": [input_shape],
            "test_data_types": [[dtype]],
            "opset_version": [11, 15],
            "input_spec_shape": [],
            "axis": axis,
            "largest": largest,
            'sorted': sortedVal,
            'k': k
        }

        models = Net(config)

        return (config, models)

    def test(self):
        self.run_and_statis(
            max_examples=30,
            reproduce=reproduce_failure('6.27.3', b'AAEAAAAAAAAB'))


if __name__ == "__main__":
    unittest.main()
