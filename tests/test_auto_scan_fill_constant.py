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
from onnxbase import randtool
import numpy as np
import unittest
import paddle


class Net(BaseNet):
    def forward(self):
        shape = self.config["shape"]
        fill_value = self.config['fill_value']
        if self.config['is_tensor']:
            fill_value = paddle.to_tensor(fill_value)
        if self.config['is_shape_tensor']:
            shape = paddle.to_tensor(shape)
        dtype = self.config["dtype"]
        x = paddle.full(shape=shape, fill_value=fill_value, dtype=dtype)
        return x


class TestFullConvert(OPConvertAutoScanTest):
    """
    api: paddle.full
    OPset version:
    """

    def sample_convert_config(self, draw):
        input_shape = draw(
            st.lists(
                st.integers(
                    min_value=2, max_value=20), min_size=1, max_size=4))
        dtype = draw(st.sampled_from(["float32", "float64", "int32", "int64"]))
        fill_value = draw(st.integers(min_value=1, max_value=5))
        # todo tensor is not supported
        is_tensor = False  # draw(st.booleans())
        is_shape_tensor = draw(st.booleans())
        config = {
            "op_names": ["fill_constant"],
            "test_data_shapes": [],
            "test_data_types": [],
            "opset_version": [9, 15],
            "input_spec_shape": [],
            "shape": input_shape,
            "dtype": dtype,
            "fill_value": fill_value,
            "is_tensor": is_tensor,
            "is_shape_tensor": is_shape_tensor,
        }

        model = Net(config)

        return (config, model)

    def test(self):
        self.run_and_statis(max_examples=30, max_duration=-1)


if __name__ == "__main__":
    unittest.main()
