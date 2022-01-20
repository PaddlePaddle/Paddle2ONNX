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
    "tile": paddle.tile,
    "expand": paddle.fluid.layers.expand,
}

opset_version_map = {
    "tile": [11, 15],
    "expand": [11, 15],
}


class Net(BaseNet):
    """
    simple Net
    """

    def forward(self, inputs):
        """
        forward
        """
        repeat_times = self.config['repeat_times']
        if self.config['repeat_times_dtype'] == "list":
            repeat_times = repeat_times
        elif self.config['repeat_times_dtype'] == "Tensor":
            repeat_times = paddle.to_tensor(repeat_times)
            repeat_times = repeat_times.astype(self.config['tesnor_dtype'])

        # x = paddle.tile(inputs, repeat_times=repeat_times)
        x = op_api_map[self.config["op_names"]](inputs, repeat_times)
        return x


class TestTileConvert(OPConvertAutoScanTest):
    """
    api: paddle.tile
    OPset version: 11, 15
    """

    def sample_convert_config(self, draw):
        input_shape = draw(
            st.lists(
                st.integers(
                    min_value=2, max_value=5), min_size=2, max_size=5))

        dtype = draw(st.sampled_from(["float32", "float64", "int32", "int64"]))
        tesnor_dtype = draw(st.sampled_from(["int32"]))
        # when repeat_times_dtype is tensor has a bug
        repeat_times_dtype = draw(st.sampled_from(["list", "Tensor"]))
        config = {
            "op_names": ["tile"],
            "test_data_shapes": [input_shape],
            "test_data_types": [[dtype]],
            "opset_version": [11, 15],
            "input_spec_shape": [],
            "repeat_times_dtype": repeat_times_dtype,
            "repeat_times": input_shape,
            "tesnor_dtype": tesnor_dtype,
        }

        models = list()
        op_names = list()
        opset_versions = list()
        for op_name, i in op_api_map.items():
            config["op_names"] = op_name
            models.append(Net(config))
            op_names.append(op_name)
        for op_name, i in op_api_map.items():
            opset_versions.append(opset_version_map[op_name])
        config["op_names"] = op_names
        config["opset_version"] = opset_versions

        return (config, models)

    def test(self):
        self.run_and_statis(max_examples=30)


class Net1(BaseNet):
    """
    simple Net
    """

    def forward(self, inputs):
        """
        forward
        """
        repeat_times = [2, paddle.to_tensor(2), 2]
        x = op_api_map[self.config["op_names"]](inputs, repeat_times)
        return x


class TestTileConvert1(OPConvertAutoScanTest):
    """
    api: paddle.tile
    OPset version: 11, 15
    """

    def sample_convert_config(self, draw):
        input_shape = draw(
            st.lists(
                st.integers(
                    min_value=2, max_value=5), min_size=2, max_size=5))

        dtype = draw(st.sampled_from(["float32", "float64", "int32", "int64"]))
        tesnor_dtype = draw(st.sampled_from(["int32"]))
        # when repeat_times_dtype is tensor has a bug
        repeat_times_dtype = draw(st.sampled_from(["list", "Tensor"]))
        config = {
            "op_names": ["tile"],
            "test_data_shapes": [input_shape],
            "test_data_types": [[dtype]],
            "opset_version": [11, 15],
            "input_spec_shape": [],
            "repeat_times_dtype": repeat_times_dtype,
            "repeat_times": input_shape,
            "tesnor_dtype": tesnor_dtype,
        }

        models = list()
        op_names = list()
        opset_versions = list()
        for op_name, i in op_api_map.items():
            config["op_names"] = op_name
            models.append(Net(config))
            op_names.append(op_name)
        for op_name, i in op_api_map.items():
            opset_versions.append(opset_version_map[op_name])
        config["op_names"] = op_names
        config["opset_version"] = opset_versions

        return (config, models)

    def test(self):
        self.run_and_statis(max_examples=30)


if __name__ == "__main__":
    unittest.main()
