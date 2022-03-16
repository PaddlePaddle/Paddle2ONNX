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

    def forward(self, x):
        """
        forward
        """
        if self.config['shape_type'] == "is_tensor":
            shape = [2, 3, 4, 5]
        elif self.config['shape_type'] == 'is_tensor_list':
            shape = [2, 3, paddle.to_tensor(np.array([4]).astype("int32")), 5]
        elif self.config['shape_type'] == 'is_list':
            shape = [2, 3, 4, 5]
        if self.config['value_is_tensor']:
            value = paddle.to_tensor(np.array([2.3]).astype("float32"))
        else:
            value = 2.3
        y = x.astype(self.config['x_dtype'])
        return y + paddle.full(shape, value, dtype=self.config['x_dtype'])


class TestFillConstantConvert(OPConvertAutoScanTest):
    """
    api: paddle.full
    OPset version: 9~15
    """

    def sample_convert_config(self, draw):
        input_shape = [2, 3, 4, 5]
        x_dtype = draw(
            st.sampled_from(["int32", "int64", "float32", "float64"]))
        shape_type = draw(
            st.sampled_from(["is_tensor", "is_tensor_list", "is_list"]))
        value_is_tensor = draw(st.booleans())

        config = {
            "op_names": ["fill_constant"],
            "test_data_shapes": [input_shape],
            "test_data_types": [["float32"]],
            "opset_version": [9, 15],
            "input_spec_shape": [[-1, -1, -1, -1]],
            "x_dtype": x_dtype,
            "shape_type": shape_type,
            "value_is_tensor": value_is_tensor
        }

        models = Net(config)

        return (config, models)

    def test(self):
        self.run_and_statis(max_examples=300)


if __name__ == "__main__":
    unittest.main()
