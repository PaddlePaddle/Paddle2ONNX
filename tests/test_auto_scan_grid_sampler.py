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

    def forward(self, x, grid):
        """
        forward
        """
        out = paddle.nn.functional.grid_sample(
            x,
            grid,
            align_corners=self.config["align_corners"],
            padding_mode=self.config["padding_mode"],
            mode=self.config["mode"],
        )
        return out


class TestGroupNormConvert(OPConvertAutoScanTest):
    """
    api: paddle.fluid.layers.nn.group_norm
    OPset version: 7, 9, 15
    """

    def sample_convert_config(self, draw):
        input_shape = draw(
            st.lists(st.integers(min_value=4, max_value=10), min_size=4, max_size=4)
        )

        grid_shape = draw(
            st.lists(st.integers(min_value=4, max_value=10), min_size=4, max_size=4)
        )

        grid_shape[0] = input_shape[0]
        grid_shape[-1] = 2
        align_corners = draw(st.booleans())
        padding_mode = draw(st.sampled_from(["reflection", "border", "zeros"]))
        mode = draw(st.sampled_from(["nearest", "bilinear"]))
        dtype = draw(st.sampled_from(["float32", "float64"]))

        config = {
            "op_names": ["grid_sample"],
            "test_data_shapes": [input_shape, grid_shape],
            "test_data_types": [[dtype], [dtype]],
            "opset_version": [16],
            "input_spec_shape": [],
            "align_corners": align_corners,
            "padding_mode": padding_mode,
            "mode": mode,
        }

        models = Net(config)

        return (config, models)

    @_test_with_pir
    def test(self):
        self.run_and_statis(max_examples=30)


if __name__ == "__main__":
    unittest.main()
