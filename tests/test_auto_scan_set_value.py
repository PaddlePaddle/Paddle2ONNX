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
import copy
from onnxbase import _test_with_pir


# TODO(wangmingkai02): add test for set_value which none_axes_ > 0
class Net(BaseNet):
    """
    simple Net
    """

    def forward(self, inputs, update_input):
        """
        forward
        """
        x = inputs + 1
        x[3:] = 3
        x[2:] = update_input
        return x


class TestSetValueConvert(OPConvertAutoScanTest):
    """
    api: set_value
    OPset version: 17, 19
    """

    def sample_convert_config(self, draw):
        input_shape = draw(
            st.lists(st.integers(min_value=5, max_value=20), min_size=1, max_size=4)
        )

        update_input_shape = copy.copy(input_shape)
        update_input_shape[0] = update_input_shape[0] - 2

        dtype = draw(st.sampled_from(["float32"]))
        config = {
            "op_names": ["set_value"],
            "test_data_shapes": [input_shape, update_input_shape],
            "test_data_types": [[dtype], [dtype]],
            "opset_version": [17, 19],
            "input_spec_shape": [input_shape, update_input_shape],
        }

        models = Net(config)

        return (config, models)

    @_test_with_pir
    def test(self):
        self.run_and_statis(max_examples=30)


if __name__ == "__main__":
    unittest.main()
