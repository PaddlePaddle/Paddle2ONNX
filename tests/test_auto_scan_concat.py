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

    def forward(self, inputs1, inputs2):
        """
        forward
        """
        axis = self.config['axis']
        if self.config['isTensor']:
            axis = paddle.to_tensor(axis, dtype=self.config['axis_dtype'])
        x = paddle.concat([inputs1, inputs2], axis=axis)
        return x


class TestConcatConvert(OPConvertAutoScanTest):
    """
    api: paddle.concat
    OPset version: 7, 9, 15
    """

    def sample_convert_config(self, draw):
        input_shape = draw(
            st.lists(
                st.integers(
                    min_value=4, max_value=8), min_size=2, max_size=5))
        axis_dtype = "int64"  # 只能设置为INT64，设置为INT32时会在axis_tensor后增加cast导致取不到constant数值
        dtype = draw(
            st.sampled_from(
                ["float16", "float32", "float64", "int32", "int64"]))

        axis = draw(
            st.integers(
                min_value=-len(input_shape), max_value=len(input_shape) - 1))

        isTensor = draw(st.booleans())
        config = {
            "op_names": ["concat"],
            "test_data_shapes": [input_shape, input_shape],
            "test_data_types": [[dtype], [dtype]],
            "opset_version": [7, 9, 15],
            "input_spec_shape": [],
            "axis": axis,
            "axis_dtype": axis_dtype,
            "isTensor": isTensor,
        }

        models = Net(config)

        return (config, models)

    def test(self):
        self.run_and_statis(max_examples=30)


if __name__ == "__main__":
    unittest.main()
