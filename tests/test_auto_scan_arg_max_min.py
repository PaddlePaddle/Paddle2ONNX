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
    "arg_max": paddle.argmax,
    "arg_min": paddle.argmin,
}

## TODO: support for 15
opset_version_map = {
    "arg_max": [7, 11, 12, 13, 14],
    "arg_min": [7, 11, 12, 13, 14],
}


class Net(BaseNet):
    def forward(self, inputs):
        return op_api_map[self.config["op_names"]](
            inputs,
            axis=self.config["axis"],
            keepdim=self.config["keepdim"],
            dtype=self.config["dtype"])


class TestArgOPConvert(OPConvertAutoScanTest):
    """Testcases for all the unary operators."""

    def sample_convert_config(self, draw):
        input_shape = draw(
            st.lists(
                st.integers(
                    min_value=2, max_value=20), min_size=1, max_size=4))

        data_shapes = input_shape
        input_specs = [-1] * len(input_shape)
        dtype = draw(st.sampled_from(["float32", "int32"]))
        output_dtype = draw(st.sampled_from(["int32", "int64"]))
        axis = draw(
            st.integers(
                min_value=-len(input_shape) + 1, max_value=len(input_shape) -
                1))
        keepdim = draw(st.booleans())
        config = {
            "op_names": "",
            "test_data_shapes": [data_shapes],
            "test_data_types": [[dtype]],
            "opset_version": [7, 9, 15],
            "input_spec_shape": [input_specs],
            "keepdim": keepdim,
            "axis": axis,
            "dtype": output_dtype,
        }
        models = list()
        op_names = list()
        opset_versions = list()
        for op_name, i in op_api_map.items():
            config["op_names"] = op_name
            models.append(Net(config))
            op_names.append(op_name)
        for op_name, vs in opset_version_map.items():
            opset_versions.append(vs)
        config["op_names"] = op_names
        config["opset_version"] = opset_versions
        return (config, models)

    def test(self):
        self.run_and_statis(max_examples=40)


if __name__ == "__main__":
    unittest.main()
