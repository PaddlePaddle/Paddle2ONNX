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


class Net(BaseNet):
    """
    simple Net
    """

    def forward(self, inputs):
        """
        forward
        """
        x = paddle.multinomial(
            inputs,
            num_samples=self.config["num_samples"],
            replacement=self.config["replacement"])
        return x


class TestMultinomialConvert(OPConvertAutoScanTest):
    """
    api: paddle.multinomial
    Opset version: 7, 9, 15
    """

    def sample_convert_config(self, draw):
        input_shape = draw(
            st.lists(
                st.integers(
                    min_value=1, max_value=10), min_size=1, max_size=2))

        replacement = draw(st.booleans())
        if replacement:
            num_samples = draw(
                st.integers(
                    min_value=1, max_value=input_shape[-1]))
        else:
            num_samples = 1

        dtype = draw(st.sampled_from(["float32", "float64"]))

        def generator_data():
            import random
            import numpy as np
            input_data = np.random.random_sample(input_shape)
            return input_data

        config = {
            "op_names": ["multinomial"],
            "test_data_shapes": [generator_data],
            "test_data_types": [[dtype]],
            "opset_version": [7, 9, 15],
            "num_samples": num_samples,
            "replacement": replacement,
            "input_spec_shape": [],
        }

        models = Net(config)

        return (config, models)

    def test(self):
        self.run_and_statis(max_examples=30)


if __name__ == "__main__":
    unittest.main()
