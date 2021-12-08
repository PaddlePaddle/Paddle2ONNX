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

op_api_map = {
    "relu": paddle.nn.functional.relu,
    "sigmoid": paddle.nn.functional.sigmoid
}


class Net(BaseNet):
    def forward(self, inputs):
        return op_api_map[self.others_list[0]](inputs)


class TestUnaryOPConvert(OPConvertAutoScanTest):
    """Testcases for all the unary operators."""

    def sample_convert_config(self, draw):
        input_shape = draw(
            st.lists(
                st.integers(
                    min_value=2, max_value=20), min_size=4, max_size=4))

        data_shapes = input_shape
        input_specs = [-1, input_shape[1], -1, -1]
        config = {
            "op_names": "",
            "test_data_shapes": [data_shapes],
            "test_data_types": [['float32']],
            "opset_version": [9],
            "input_spec_shape": [input_specs],
        }
        models = list()
        op_names = list()
        for op_name, i in op_api_map.items():
            models.append(Net(config, op_name))
            op_names.append(op_name)
        config["op_names"] = op_names
        return (config, models)

    def test(self):
        self.run_and_statis(max_examples=40)


if __name__ == "__main__":
    unittest.main()
