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

op_api_map = {
    "reduce_all": paddle.all,
    "reduce_any": paddle.any,
}

opset_version_map = {
    "reduce_all": [11, 12, 13, 18],
    "reduce_any": [11, 12, 13, 18],
}


class Net(BaseNet):
    """
    simple Net
    """

    def forward(self, inputs):
        """
        forward
        """
        x = op_api_map[self.config["op_names"]](inputs,
                                                axis=self.config["dim"],
                                                keepdim=self.config["keep_dim"])
        x = paddle.unsqueeze(x, axis=0)
        x = x.astype("int32")
        return x


class TestReduceAllConvert(OPConvertAutoScanTest):
    """
    api: paddle.fluid.layers.reduce_all
    OPset version: 7, 9, 15
    """

    def sample_convert_config(self, draw):
        input_shape = draw(
            st.lists(
                st.integers(
                    min_value=2, max_value=20), min_size=1, max_size=5))

        input_spec = [-1] * len(input_shape)

        dtype = draw(st.sampled_from(["bool"]))
        axis_type = draw(st.sampled_from([
            "list",
            "int",
        ]))
        if axis_type == "int":
            dim = draw(
                st.integers(
                    min_value=-len(input_shape), max_value=len(input_shape) -
                    1))
        elif axis_type == "list":
            lenSize = random.randint(1, len(input_shape))
            dim = []
            for i in range(lenSize):
                dim.append(random.choice([i, i - len(input_shape)]))
        keep_dim = draw(st.booleans())

        config = {
            "op_names": ["reduce_all"],
            "test_data_shapes": [input_shape],
            "test_data_types": [[dtype]],
            "opset_version": [7, 13, 18],
            "dim": dim,
            "keep_dim": keep_dim,
            "input_spec_shape": []
        }

        models = list()
        op_names = list()
        opset_versions = list()
        for op_name, i in op_api_map.items():
            config["op_names"] = op_name
            models.append(Net(config))
            op_names.append(op_name)
            opset_versions.append(opset_version_map[op_name])
        config["op_names"] = op_names
        config["opset_version"] = opset_versions
        return (config, models)

    def test(self):
        self.run_and_statis(max_examples=30, max_duration=-1)


if __name__ == "__main__":
    unittest.main()
