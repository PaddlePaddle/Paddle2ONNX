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
    'bilinear': 'bilinear_interp_v2',
    'trilinear': 'trilinear_interp_v2',
    'nearest': 'nearest_interp_v2',
    'bicubic': 'bicubic_interp_v2',
    'nearest_v1': 'nearest_interp',
    'bilinear_v1': 'bilinear_interp',
}

data_format_map = {
    'linear': 'NCW',
    'bilinear': 'NCHW',
    'trilinear': 'NCDHW',
    'nearest': 'NCHW',
    'bicubic': 'NCHW',
}

op_set_map = {
    'linear': [9, 10, 11, 12, 13, 14, 15],
    'bilinear': [9, 10, 11, 12, 13, 14, 15],
    'trilinear': [9, 10, 11, 12, 13, 14, 15],
    'nearest': [9, 10, 11, 12, 13, 14, 15],
    'bicubic': [11, 12, 13, 14, 15],
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
        if self.config['is_scale_tensor'] and scale_factor is not None:
            scale_factor = paddle.to_tensor(
                scale_factor, dtype=self.config['scale_dtype'])
        size = self.config['size']
        if self.config['is_size_tensor'] and size is not None:
            size = paddle.to_tensor(size, self.config['size_dtype'])

        align_mode = self.config['align_mode']
        mode = self.config['mode']
        align_corners = self.config['align_corners'][0]
        data_format = self.config['data_format']
        # align_corners True is only set with the interpolating modes: linear | bilinear | bicubic | trilinear
        if mode == "nearest":
            align_corners = False
        x = paddle.nn.functional.interpolate(
            x=inputs,
            size=size,
            scale_factor=scale_factor,
            mode=mode,
            align_corners=align_corners,
            align_mode=align_mode,
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
        size_dtype = draw(st.sampled_from(["int32", "int64"]))
        scale_dtype = draw(st.sampled_from(["float32", "float64"]))
        # mode = draw(st.sampled_from(["linear"]))
        # mode = draw(st.sampled_from(["nearest"]))
        # mode = draw(st.sampled_from(["bilinear"]))
        # mode = draw(st.sampled_from(["bicubic"]))
        # mode = draw(st.sampled_from(["trilinear"]))
        mode = draw(
            st.sampled_from(
                ["linear", "nearest", "bilinear", "bicubic", "trilinear"]))
        align_corners = draw(st.booleans()),
        align_mode = draw(st.integers(min_value=0, max_value=1))
        data_format = data_format_map[mode]
        if data_format == "NCW":
            num = 1
            input_shape = np.random.choice(input_shape, 3)
            input_shape[0] = 1  # there is a bug when index > 1
        elif data_format == "NCHW":
            num = 2
            input_shape = np.random.choice(input_shape, 4)
        else:
            num = 3
            input_shape = np.random.choice(input_shape, 5)

        is_scale_tensor = False
        is_size_tensor = False
        if draw(st.booleans()):
            size = None
            if draw(st.booleans()):
                # float
                scale_factor = draw(st.floats(min_value=1.2, max_value=2.0))
            else:
                # list
                is_scale_tensor = draw(st.booleans())
                scale_factor = draw(
                    st.lists(
                        st.floats(
                            min_value=1.2, max_value=2.0),
                        min_size=num,
                        max_size=num))
        else:
            scale_factor = None
            # list
            is_size_tensor = draw(st.booleans())
            size = draw(
                st.lists(
                    st.integers(
                        min_value=12, max_value=30),
                    min_size=num,
                    max_size=num))

        op_name = op_api_map[mode]
        opset_version = op_set_map[mode]
        if mode in ["linear", "bilinear", "trilinear"]:
            if not align_corners[0] and align_mode == 1:
                opset_version = [9, 10, 11, 12, 13, 14, 15]
            else:
                opset_version = [11, 12, 13, 14, 15]
        config = {
            "op_names": [op_name],
            "test_data_shapes": [input_shape],
            "test_data_types": [[dtype]],
            "opset_version": opset_version,
            "input_spec_shape": [],
            "size": size,
            "scale_factor": scale_factor,
            "mode": mode,
            "align_corners": align_corners,
            "align_mode": align_mode,
            "data_format": data_format,
            "is_scale_tensor": is_scale_tensor,
            "is_size_tensor": is_size_tensor,
            "size_dtype": size_dtype,
            "scale_dtype": scale_dtype,
        }

        models = Net(config)

        return (config, models)

    def test(self):
        self.run_and_statis(max_examples=100)


if __name__ == "__main__":
    unittest.main()
