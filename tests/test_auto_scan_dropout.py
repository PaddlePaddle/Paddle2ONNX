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
import random
from onnxbase import _test_only_pir


class Net(BaseNet):
    """
    simple Net
    """

    def forward(self, x):
        """
        forward
        """
        if self.config["tensor_attr"]:
            p = paddle.to_tensor(self.config["p"], dtype="float32")
        else:
            p = self.config["p"]
        # when training is true, has diff
        x = paddle.nn.functional.dropout(
            x, training=False, p=p, axis=self.config["axis"], mode=self.config["mode"]
        )
        return x


class TestDropoutConvert(OPConvertAutoScanTest):
    """
    api: paddle.nn.functional.dropout
    OPset version: 7, 9, 15
    """

    def sample_convert_config(self, draw):
        input_shape = draw(
            st.lists(st.integers(min_value=2, max_value=8), min_size=0, max_size=5)
        )
        # "float64" has a bug
        dtype = draw(st.sampled_from(["float32"]))
        p = random.random()
        mode = draw(st.sampled_from(["upscale_in_train", "downscale_in_infer"]))

        if draw(st.booleans()) or len(input_shape) == 0:
            axis = None
        else:
            axis = draw(st.integers(min_value=0, max_value=len(input_shape) - 1))

        tensor_attr = draw(st.booleans())

        if len(input_shape) == 0:
            axis = 0
            tensor_attr = False  # must be false when 0D tensor

        if len(input_shape) == 1:
            axis = 0

        config = {
            "op_names": ["dropout"],
            "test_data_shapes": [input_shape],
            "test_data_types": [[dtype]],
            "opset_version": [7, 9, 15],
            "input_spec_shape": [],
            "axis": axis,
            "mode": mode,
            "p": p,
            "tensor_attr": tensor_attr,
        }
        if axis is not None:
            if mode in ["upscale_in_train"]:
                config["op_names"] = [""]
            else:
                config["op_names"] = ["scale"]
        models = Net(config)

        return (config, models)

    @_test_only_pir
    def test(self):
        self.run_and_statis(max_examples=30)


if __name__ == "__main__":
    unittest.main()
