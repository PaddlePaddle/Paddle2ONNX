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
from onnxbase import _test_with_pir


class Net(BaseNet):
    """
    simple Net
    """

    def forward(self, inputs1, inputs2):
        """
        forward
        """
        x = paddle.stack([inputs1, inputs2], axis=self.config["axis"])
        return x


class TestStackConvert(OPConvertAutoScanTest):
    """
    api: paddle.stack
    OPset version: 7, 9, 15
    """

    def sample_convert_config(self, draw):
        input_shape = draw(
            st.lists(st.integers(min_value=4, max_value=8), min_size=0, max_size=5)
        )

        dtype = draw(st.sampled_from(["float32", "float64", "int32", "int64"]))
        if len(input_shape) > 0:
            axis = draw(
                st.integers(min_value=-len(input_shape), max_value=len(input_shape) - 1)
            )
        else:
            axis = 0

        config = {
            "op_names": ["stack"],
            "test_data_shapes": [input_shape, input_shape],
            "test_data_types": [[dtype], [dtype]],
            "opset_version": [7, 9, 15],
            "input_spec_shape": [],
            "axis": axis,
        }

        models = Net(config)

        return (config, models)

    @_test_with_pir
    def test(self):
        self.run_and_statis(max_examples=30)


if __name__ == "__main__":
    unittest.main()
