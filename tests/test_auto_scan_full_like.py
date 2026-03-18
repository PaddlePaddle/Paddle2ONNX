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
import hypothesis.strategies as st
import unittest
import paddle
from onnxbase import randtool, _test_only_pir

op_api_map = {
    "fill_any_like": paddle.ones_like,
    "fill_zeros_like": paddle.zeros_like,
}


class Net(BaseNet):
    """
    simple Net
    """

    def forward(self, x):
        """
        forward
        """
        x = op_api_map[self.config["op_names"]](x)
        x = x.astype("int32")
        return x


class TestFullLikeConvert(OPConvertAutoScanTest):
    """
    api: paddle.ones_like && paddle.zeros_like
    OPset version: 8, 13, 15
    """

    def sample_convert_config(self, draw):
        input_shape = draw(
            st.lists(st.integers(min_value=10, max_value=20), min_size=1, max_size=4)
        )

        dtype = draw(st.sampled_from(["bool", "int32", "int64", "float32", "float64"]))

        config = {
            "op_names": "",
            "test_data_shapes": [input_shape],
            "test_data_types": [[dtype]],
            "opset_version": [8, 13, 15],
            "input_spec_shape": [],
        }

        models = list()
        op_names = list()
        for op_name, i in op_api_map.items():
            config["op_names"] = op_name
            models.append(Net(config))
            op_names.append(op_name)
        config["op_names"] = op_names

        return (config, models)

    @_test_only_pir
    def test(self):
        self.run_and_statis(max_examples=30)


class Net2(BaseNet):
    """
    simple Net
    """

    def forward(self, x, fill_value):
        """
        forward
        """
        x = paddle.full_like(x, fill_value)
        x = x.astype("int32")
        return x


class TestFullLikeConvert2(OPConvertAutoScanTest):
    """
    api: paddle.full_like
    OPset version: 8, 13, 15
    """

    def sample_convert_config(self, draw):
        input_shape = draw(
            st.lists(st.integers(min_value=10, max_value=20), min_size=1, max_size=4)
        )

        dtype = draw(st.sampled_from(["bool", "int32", "int64", "float32", "float64"]))

        def gen_fill_value():
            return randtool("float", 0, 1, [1])

        config = {
            "op_names": ["full_like"],
            "test_data_shapes": [input_shape, gen_fill_value],
            "test_data_types": [[dtype], [dtype]],
            "opset_version": [8, 13, 15],
            "input_spec_shape": [],
        }

        models = Net2(config)

        return (config, models)

    @_test_only_pir
    def test(self):
        self.run_and_statis(max_examples=30)


if __name__ == "__main__":
    unittest.main()
