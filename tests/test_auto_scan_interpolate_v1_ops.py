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
    'NEAREST': 'nearest_interp',
    'BILINEAR': 'bilinear_interp',
}

data_format_map = {
    'NEAREST': 'NCHW',
    'BILINEAR': 'NCHW',
}

op_set_map = {
    'NEAREST': [11, 12, 13, 14, 15],
    'BILINEAR': [11, 12, 13, 14, 15],
}


class Net1(BaseNet):
    """
    simple Net
    """

    def forward(self, inputs):
        """
        forward
        """
        align_corners = self.config['align_corners'][0]
        align_mode = self.config['align_mode']
        out_shape = self.config['size']
        scale = self.config['scale_factor']
        data_format = self.config['data_format']
        mode = self.config['mode']

        x = paddle.fluid.layers.image_resize(
            inputs,
            out_shape=out_shape,
            scale=scale,
            resample=mode,
            align_corners=align_corners,
            align_mode=align_mode,
            data_format=data_format)
        return x


class TestInterpolateConvert1(OPConvertAutoScanTest):
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
        # mode = draw(st.sampled_from(["NEAREST"]))
        # mode = draw(st.sampled_from(["BILINEAR"]))
        mode = draw(st.sampled_from(["NEAREST", "BILINEAR"]))
        align_corners = draw(st.booleans()),
        align_mode = draw(st.integers(min_value=0, max_value=1))
        data_format = data_format_map[mode]
        input_shape = np.random.choice(input_shape, 4)

        if draw(st.booleans()):
            size = None
            # scale_factor should b even. eg [2, 4, 6, 8, 10]
            scale_factor = draw(st.integers(min_value=2, max_value=10))
            scale_factor = scale_factor + 1 if scale_factor % 2 != 0 else scale_factor
        else:
            scale_factor = None
            size1 = draw(st.integers(min_value=12, max_value=30))
            # NEAREST, size should be even
            if mode == 'NEAREST':
                size1 = size1 + 1 if size1 % 2 != 0 else size1
            size2 = draw(st.integers(min_value=12, max_value=30))
            if mode == 'NEAREST':
                size2 = size2 + 1 if size2 % 2 != 0 else size2
            size = [size1, size2]

        op_name = op_api_map[mode]
        opset_version = op_set_map[mode]
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
        }

        models = Net1(config)

        return (config, models)

    def test(self):
        self.run_and_statis(max_examples=30)


if __name__ == "__main__":
    unittest.main()
