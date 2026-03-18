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
import numpy as np
import unittest
import paddle
import random
from random import shuffle
from onnxbase import _test_only_pir


class Net(BaseNet):
    """
    simple Net
    """

    def forward(self, input):
        """
        forward
        """
        k = self.config["k"]
        if self.config["isTensor"]:
            k = paddle.to_tensor(k, dtype=self.config["k_dtype"])
        x = paddle.topk(
            input,
            k=k,
            axis=self.config["axis"],
            largest=self.config["largest"],
            sorted=self.config["sorted"],
        )
        return x


class TestTopkv2Convert(OPConvertAutoScanTest):
    """
    api: paddle.topk
    OPset version: 11, 15
    """

    def sample_convert_config(self, draw):
        input_shape = draw(
            st.lists(st.integers(min_value=1, max_value=3), min_size=0, max_size=5)
        )
        axis = None
        if draw(st.booleans()) and len(input_shape) > 0:
            axis = draw(
                st.integers(min_value=-len(input_shape), max_value=len(input_shape) - 1)
            )

        dtype = draw(st.sampled_from(["float32", "float64", "int32", "int64"]))
        k_dtype = draw(st.sampled_from(["int32", "int64"]))
        largest = draw(st.booleans())
        # has a diff when sorted is False
        sorted = draw(st.booleans())

        def generator_data():
            if len(input_shape) == 0:
                return np.array(10, dtype="float32")
            prod = np.prod(input_shape)
            input_data = np.array(list(range(0, prod)))
            shuffle(input_data)
            input_data = input_data.reshape(input_shape)
            return input_data

        if len(input_shape) > 0:
            if axis is not None:
                k = random.randint(1, input_shape[axis])
            else:
                k = random.randint(1, input_shape[-1])
        else:
            k = 1
            axis = 0
        isTensor = draw(st.booleans())

        config = {
            "op_names": ["top_k_v2"],
            "test_data_shapes": [generator_data],
            "test_data_types": [[dtype]],
            "opset_version": [11, 12, 13, 14, 15],
            "input_spec_shape": [],
            "axis": axis,
            "largest": largest,
            "sorted": sorted,
            "isTensor": isTensor,
            "k": k,
            "k_dtype": k_dtype,
            "rtol": 200,
            "delta": 200,
            "input_shape": input_shape,
        }

        models = Net(config)

        return (config, models)

    @_test_only_pir
    def test(self):
        self.run_and_statis(max_examples=80)


if __name__ == "__main__":
    unittest.main()
