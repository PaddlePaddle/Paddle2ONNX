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
        x = paddle.fluid.layers.reduce_prod(
            inputs, dim=self.config["dim"], keep_dim=self.config["keep_dim"])
        return x


class TestReduceProdConvert(OPConvertAutoScanTest):
    """
    api: paddle.fluid.layers.reduce_prod
    OPset version: 7, 9, 15
    """

    def sample_convert_config(self, draw):
        input_shape = draw(
            st.lists(
                st.integers(
                    min_value=20, max_value=100),
                min_size=4,
                max_size=4))

        input_spec = [-1] * len(input_shape)
        # onnxruntime is not supported "float64"
        dtype = draw(st.sampled_from(["float32"]))
        axis_type = draw(st.sampled_from([
            "list",
            "int",
        ]))
        if axis_type == "int":
            dim = draw(st.integers(min_value=0, max_value=3))
        elif axis_type == "list":
            dim = np.array(
                draw(
                    st.lists(
                        st.integers(
                            min_value=0, max_value=3),
                        min_size=1,
                        max_size=2))).tolist()
            if len(dim) > len(set(dim)):
                dim[0] = dim[0] - 1

        keep_dim = draw(st.integers(min_value=0, max_value=1))
        keep_dim = bool(keep_dim)
        config = {
            "op_names": ["reduce_prod"],
            "test_data_shapes": [input_shape],
            "test_data_types": [[dtype]],
            "opset_version": [7, 9, 15],
            "dim": dim,
            "keep_dim": keep_dim,
            "input_spec_shape": []
        }

        models = Net(config)

        return (config, models)

    def test(self):
        self.run_and_statis(max_examples=100)


if __name__ == "__main__":
    unittest.main()
