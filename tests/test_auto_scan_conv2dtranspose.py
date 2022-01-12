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

    def forward(self, inputs, weight):
        """
        forward
        """
        x = paddle.nn.functional.conv2d_transpose(
            inputs,
            weight,
            bias=None,
            stride=self.config["stride"],
            padding=self.config["padding"],
            output_padding=self.config["output_padding"],
            dilation=self.config["dilation"],
            groups=self.config["groups"],
            output_size=None,
            data_format=self.config["data_format"])

        return x


class TestConv2dTransposeConvert(OPConvertAutoScanTest):
    """
    api: paddle.nn.functional.conv2d_transpose
    OPset version: 7, 9, 15
    1.OPset version需要根据op_mapper中定义的version来设置。
    2.测试中所有OP对应升级到Opset version 15。
    """

    def sample_convert_config(self, draw):
        input_shape = draw(
            st.lists(
                st.integers(
                    min_value=20, max_value=100),
                min_size=4,
                max_size=4))

        kernel_size = draw(
            st.lists(
                st.integers(
                    min_value=1, max_value=7), min_size=4, max_size=4))

        data_format = draw(st.sampled_from(["NCHW", "NHWC"]))
        data_format = "NCHW"

        groups = draw(st.integers(min_value=1, max_value=4))
        muti1 = draw(st.integers(min_value=1, max_value=4))

        # kernel_size: [Cin, Cout, Hf, Wf]
        # the channel of input must be divisible by groups
        kernel_size[0] = kernel_size[0] * groups
        kernel_size[1] = groups * muti1
        # input: [N, Cin, Hin, Win]
        input_shape[1] = kernel_size[0]

        strides = draw(
            st.lists(
                st.integers(
                    min_value=1, max_value=5), min_size=1, max_size=2))
        if len(strides) == 1:
            strides = strides[0]

        padding_type = draw(st.sampled_from(["str", "list", "int", "tuple"]))
        padding = None
        if padding_type == "str":
            padding = draw(st.sampled_from(["SAME", "VALID"]))
        elif padding_type == "int":
            padding = draw(st.integers(min_value=1, max_value=5))
        elif padding_type == "tuple":
            padding1 = np.expand_dims(
                np.array(
                    draw(
                        st.lists(
                            st.integers(
                                min_value=1, max_value=5),
                            min_size=2,
                            max_size=2))),
                axis=0).tolist()
            padding2 = np.expand_dims(
                np.array(
                    draw(
                        st.lists(
                            st.integers(
                                min_value=1, max_value=5),
                            min_size=2,
                            max_size=2))),
                axis=0).tolist()
            if data_format == "NCHW":
                padding = [[0, 0]] + [[0, 0]] + padding1 + padding2
            else:
                padding = [[0, 0]] + padding1 + padding2 + [[0, 0]]
        elif padding_type == "list":
            if draw(st.booleans()):
                padding = draw(
                    st.lists(
                        st.integers(
                            min_value=1, max_value=5),
                        min_size=2,
                        max_size=2))
            else:
                padding = draw(
                    st.lists(
                        st.integers(
                            min_value=1, max_value=5),
                        min_size=4,
                        max_size=4))

        dilations = draw(
            st.lists(
                st.integers(
                    min_value=1, max_value=3), min_size=1, max_size=2))
        if len(dilations) == 1:
            dilations = dilations[0]
        if padding == "SAME":
            dilations = 1

        dtype = draw(st.sampled_from(["float32"]))

        config = {
            "op_names": ["conv2d_transpose"],
            "test_data_shapes": [input_shape, kernel_size],
            "test_data_types": [[dtype], [dtype]],
            "opset_version": [7, 9, 15],
            "input_spec_shape":
            [input_shape, kernel_size],  # [-1, input_shape[1], -1, -1]
            "data_format": data_format,
            "stride": strides,
            "dilation": dilations,
            "padding": padding,
            "output_padding": 0,
            "groups": groups,
            "input_shape": input_shape,
            "kernel_size": kernel_size,
            "delta": 1e-4,
            "rtol": 1e-4
        }

        models = Net(config)

        return (config, models)

    def test(self):
        self.run_and_statis(max_examples=30)


if __name__ == "__main__":
    unittest.main()
