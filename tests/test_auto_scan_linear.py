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
    simple Net for nn.Linear.
    """

    def __init__(self, config=None):
        super(Net, self).__init__(config)
        in_features = self.config["in_features"]
        out_features = self.config["out_features"]
        self.fc = paddle.nn.Linear(
            in_features=in_features,
            out_features=out_features,
            bias_attr=None if self.config["has_bias"] else False,
        )

    def forward(self, inputs):
        x = self.fc(inputs)
        return x


class TestLinearConvert(OPConvertAutoScanTest):
    """
    Test nn.Linear export in PIR format.
    """

    def sample_convert_config(self, draw):
        input_shape = draw(
            st.lists(
                st.integers(min_value=1, max_value=32),
                min_size=2,
                max_size=4,
                unique=False,
            )
        )
        in_features = draw(st.integers(min_value=16, max_value=256))
        out_features = draw(st.integers(min_value=16, max_value=256))
        input_shape[-1] = in_features

        config = {
            "in_features": in_features,
            "out_features": out_features,
            "has_bias": draw(st.booleans()),
            "op_names": ["linear_v2"],
            "test_data_shape": input_shape,
            "use_gpu": True,
            "min_opset_version": 7,
            "max_opset_version": 15,
        }

        models = Net(config)

        return (config, models)

    @_test_with_pir
    def test(self):
        self.run_and_statis(max_examples=30)


if __name__ == "__main__":
    unittest.main()
