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

op_api_map = {
    'linear': 'linear_interp_v2',
    # 'bilinear_interp_v2': 'linear',
    'trilinear': 'trilinear_interp_v2',
    'nearest': 'nearest_interp_v2',
    # 'bicubic_interp_v2': 'cubic',
    # 'bilinear_interp': 'linear',
    # 'nearest_interp': 'nearest',
}


class Net(BaseNet):
    """
    simple Net
    """

    def forward(self, inputs):
        """
        forward
        """
        scale_factor = self.config['scale_factor']
        size = self.config['size']
        align_mode = self.config['align_mode']
        mode = self.config['mode']
        align_corners = self.config['align_corners']
        data_format = self.config['data_format']
        if mode == "nearest":
            align_corners = False
        elif mode == "linear":
            align_corners = False
            align_mode = 1
        else:
            align_corners = False
            align_mode = 1
        x = paddle.nn.functional.interpolate(
            x=inputs,
            size=size,
            scale_factor=scale_factor,
            mode=mode,
            align_corners=False,
            align_mode=1,
            data_format=data_format)
        return x


class TestInterpolateConvert(OPConvertAutoScanTest):
    """
    api: paddle.nn.functional.interpolate
    OPset version: 9, 11, 15
    """

    def sample_convert_config(self, draw):
        input_shape = draw(
            st.lists(
                st.integers(
                    min_value=2, max_value=8), min_size=5, max_size=6))

        dtype = draw(st.sampled_from(["float32"]))
        # mode = draw(st.sampled_from(["linear"]))
        # mode = draw(st.sampled_from(["nearest"]))
        mode = draw(st.sampled_from(["trilinear"]))
        align_corners = draw(st.booleans()),
        align_mode = draw(st.integers(min_value=0, max_value=1))
        # data_format = draw(st.sampled_from(["NCW"]))
        # data_format = draw(st.sampled_from(["NCHW"]))
        data_format = draw(st.sampled_from(["NCDHW"]))
        if data_format == "NCW":
            input_shape = np.random.choice(input_shape, 3)
        elif data_format == "NCHW":
            input_shape = np.random.choice(input_shape, 4)
        else:
            input_shape = np.random.choice(input_shape, 5)
        if draw(st.booleans()):
            size = None
            if draw(st.booleans()):
                # float
                scale_factor = draw(st.floats(min_value=1.2, max_value=2.0))
            else:
                # list
                scale_factor = [1.2, 1.3, 1.4, 1.5, 1.5]
                if data_format == "NCW":
                    scale_factor = np.random.choice(size, 1)
                elif data_format == "NCHW":
                    scale_factor = np.random.choice(size, 2)
                else:
                    scale_factor = np.random.choice(size, 3)
        else:
            scale_factor = None
            # list
            size = [12, 13, 14]
            if data_format == "NCW":
                size = np.random.choice(size, 1)
            elif data_format == "NCHW":
                size = np.random.choice(size, 2)
            else:
                size = np.random.choice(size, 3)
        op_name = op_api_map[mode]
        config = {
            "op_names": [op_name],
            "test_data_shapes": [input_shape],
            "test_data_types": [[dtype]],
            "opset_version": [9],
            "input_spec_shape": [],
            "size": size,
            "scale_factor": scale_factor,
            "mode": mode,
            "align_corners": align_corners,
            "align_mode": align_mode,
            "data_format": data_format,
        }

        models = Net(config)

        return (config, models)

    def test(self):
        self.run_and_statis(max_examples=25)


if __name__ == "__main__":
    unittest.main()
