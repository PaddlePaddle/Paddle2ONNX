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


class Net(BaseNet):
    def forward(self, inputs):
        pad = self.config["pad"]
        mode = self.config["mode"]
        value = self.config["value"]
        data_format = self.config["data_format"]
        x = paddle.nn.functional.pad(
            inputs, pad=pad, mode=mode, value=value, data_format=data_format
        )
        return x


class TestPadopsConvert(OPConvertAutoScanTest):
    """
    api: pad2d
    OPset version:
    """

    def sample_convert_config(self, draw):
        input_shape = draw(
            st.lists(st.integers(min_value=10, max_value=15), min_size=3, max_size=5)
        )

        dtype = "float32"

        pad = draw(
            st.lists(
                st.integers(min_value=0, max_value=4),
                min_size=2 * len(input_shape),
                max_size=2 * len(input_shape),
            )
        )

        mode = draw(st.sampled_from(["constant"]))

        value = draw(st.floats(min_value=10, max_value=20))

        data_format = draw(
            st.sampled_from(["NCL", "NLC", "NCHW", "NHWC", "NCDHW", "NDHWC"])
        )

        config = {
            "op_names": ["pad"],
            "test_data_shapes": [input_shape],
            "test_data_types": [[dtype]],
            "opset_version": [7, 11, 15],
            "input_spec_shape": [],
            "mode": mode,
            "value": value,
            "pad": pad,
            "data_format": data_format,
        }

        model = Net(config)

        return (config, model)

    def test(self):
        self.run_and_statis(max_examples=30, max_duration=-1)


if __name__ == "__main__":
    unittest.main()
