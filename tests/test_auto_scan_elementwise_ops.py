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
from onnxbase import randtool
import unittest
import paddle
from onnxbase import _test_only_pir

op_api_map = {
    "elementwise_add": paddle.add,
    "elementwise_sub": paddle.subtract,
    "elementwise_div": paddle.divide,
    "elementwise_mul": paddle.multiply,
    "elementwise_mod": paddle.remainder,
}

opset_version_map = {
    "elementwise_add": [7, 9, 15],
    "elementwise_sub": [7, 9, 15],
    "elementwise_div": [7, 9, 15],
    "elementwise_mul": [7, 9, 15],
    "elementwise_mod": [15],
}


class Net(BaseNet):
    def forward(self, inputs1, inputs2):
        x = op_api_map[self.config["op_names"]](inputs1, inputs2)
        return x


class TestElementwiseopsConvert(OPConvertAutoScanTest):
    """
    api: elementwise ops
    OPset version: 7, 9, 15
    """

    def sample_convert_config(self, draw):
        input1_shape = draw(
            st.lists(st.integers(min_value=10, max_value=20), min_size=0, max_size=4)
        )

        if len(input1_shape) > 0:
            if draw(st.booleans()):
                # [N * N] + [N]
                input2_shape = [input1_shape[-1]]
            elif draw(st.booleans()):
                # [N * N] + [N * N]
                input2_shape = input1_shape
            else:
                # [N * N] + []
                input2_shape = []
        else:
            if draw(st.booleans()):
                # [] + []
                input2_shape = input1_shape
            else:
                # [] + [N * N]
                input2_shape = draw(
                    st.lists(
                        st.integers(min_value=10, max_value=20), min_size=1, max_size=4
                    )
                )

        dtype = draw(st.sampled_from(["float32", "int32"]))

        def generator_data():
            input_data = randtool("int", -5.0, 5.0, input2_shape)
            input_data[abs(input_data) < 1.0] = 1.0
            return input_data

        config = {
            "op_names": ["elementwise_add"],
            "test_data_shapes": [input1_shape, generator_data],
            "test_data_types": [[dtype], [dtype]],
            "opset_version": [7, 9, 15],
            "input_spec_shape": [],
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

    @_test_only_pir
    def test(self):
        self.run_and_statis(max_examples=30)


op_api_map_2 = {
    "elementwise_min": paddle.minimum,
    "elementwise_max": paddle.maximum,
    "elementwise_pow": paddle.pow,
}

opset_version_map_2 = {
    "elementwise_min": [9, 15],
    "elementwise_max": [9, 15],
    "elementwise_pow": [7, 9, 15],
}


class Net_2(BaseNet):
    def forward(self, inputs1, inputs2):
        x = op_api_map_2[self.config["op_names"]](inputs1, inputs2)
        return x


class TestElementwiseopsConvert_2(OPConvertAutoScanTest):
    """
    api: elementwise ops
    OPset version: 7, 9, 15
    """

    def sample_convert_config(self, draw):
        input1_shape = draw(
            st.lists(st.integers(min_value=10, max_value=20), min_size=0, max_size=4)
        )

        if len(input1_shape) > 0:
            if draw(st.booleans()):
                # [N * N] + [N]
                input2_shape = [input1_shape[-1]]
            elif draw(st.booleans()):
                # [N * N] + [N * N]
                input2_shape = input1_shape
            else:
                # [N * N] + []
                input2_shape = []
        else:
            if draw(st.booleans()):
                # [] + []
                input2_shape = input1_shape
            else:
                # [] + [N * N]
                input2_shape = draw(
                    st.lists(
                        st.integers(min_value=10, max_value=20), min_size=1, max_size=4
                    )
                )

        dtype = draw(st.sampled_from(["float32"]))

        config = {
            "op_names": ["elementwise_add"],
            "test_data_shapes": [input1_shape, input2_shape],
            "test_data_types": [[dtype], [dtype]],
            "opset_version": [7, 9, 16],
            "input_spec_shape": [],
        }

        models = list()
        op_names = list()
        opset_versions = list()
        for op_name, i in op_api_map_2.items():
            config["op_names"] = op_name
            models.append(Net_2(config))
            op_names.append(op_name)
        for op_name, i in op_api_map_2.items():
            opset_versions.append(opset_version_map_2[op_name])
        config["op_names"] = op_names
        config["opset_version"] = opset_versions

        return (config, models)

    @_test_only_pir
    def test(self):
        self.run_and_statis(max_examples=30)


if __name__ == "__main__":
    unittest.main()
