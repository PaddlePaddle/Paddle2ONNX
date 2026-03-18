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
from onnxbase import _test_only_pir, randtool


class Net(BaseNet):
    """
    simple Net
    """

    def forward(self, arr, indices, values):
        """
        forward
        """
        x = paddle.put_along_axis(
            arr, indices, values, axis=self.config["axis"], reduce=self.config["reduce"]
        )
        return x


class TestPutAlongAxisConvert(OPConvertAutoScanTest):
    """
    api: paddle.put_along_axis
    OPset version: 11, 16, 18
    """

    def sample_convert_config(self, draw):
        input_shape = draw(
            st.lists(st.integers(min_value=1, max_value=20), min_size=2, max_size=5)
        )
        dtype = draw(st.sampled_from(["float32", "float64"]))
        dtype2 = draw(st.sampled_from(["int32", "int64"]))
        # dtype3 = draw(st.sampled_from(["float32", "float64"]))
        axis = draw(st.integers(min_value=0, max_value=len(input_shape) - 1))
        reduce = draw(st.sampled_from(["assign", "add", "multiply", "amin", "amax"]))

        opset_version = []
        if reduce == "add" or reduce == "multiply":
            opset_version.append(16)
        elif reduce == "amin" or reduce == "amax":
            opset_version.append(18)
        else:
            opset_version.append(11)

        def generator_data():
            input_data = randtool("int", 0, input_shape[axis], input_shape)
            print("wmk" * 10)
            print(input_data.shape)
            return input_data

        config = {
            "op_names": ["put_along_axis"],
            "test_data_shapes": [input_shape, generator_data, input_shape],
            "test_data_types": [[dtype], [dtype2], [dtype]],
            "opset_version": opset_version,
            "input_spec_shape": [],
            "axis": axis,
            "reduce": reduce,
        }

        models = Net(config)

        return (config, models)

    @_test_only_pir
    def test(self):
        self.run_and_statis(max_examples=30)


if __name__ == "__main__":
    unittest.main()
