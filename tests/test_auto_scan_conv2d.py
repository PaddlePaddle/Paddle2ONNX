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

from auto_scan_test import OPConvertAutoScanTest
import numpy as np
import paddle.inference as paddle_infer
from functools import partial
from typing import Optional, List, Callable, Dict, Any, Set
import unittest
import paddle

import hypothesis
from hypothesis import given, settings, seed, example, assume, reproduce_failure
import hypothesis.strategies as st


class Net(paddle.nn.Layer):
    """
    simple Net
    """

    def __init__(self, config):
        super(Net, self).__init__()
        self.stride = config["stride"]
        self.padding = config["padding"]
        self.dilation = config["dilation"]
        self.groups = config["groups"]
        self.data_format = config["data_format"]

    def forward(self, inputs, weight):
        """
        forward
        """
        x = paddle.nn.functional.conv2d(
            inputs,
            weight,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
            data_format=self.data_format)
        return x


class TestConv2dConvert(OPConvertAutoScanTest):
    """
    api: paddle.nn.Conv2d
    OPset version: 9, 10, 11, 12 
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
        kernel_size[0] = groups * muti1
        input_shape[1] = kernel_size[1] * groups

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

        config = {
            "data_format": data_format,
            "stride": strides,
            "dilation": dilations,
            "padding": padding,
            "groups": groups,
            "input_shape": input_shape,
            "kernel_size": kernel_size,
        }

        self.model = Net(config)
        self.op_name = "conv2d"
        self.test_data_shape = [input_shape, kernel_size]
        self.input_spec_shape = [[-1, input_shape[1], -1, -1], kernel_size]
        self.test_data_type = [['float32'], ['float32']]

        return config

    def test(self):
        self.run_and_statis(max_examples=25, opset_version=[9, 10, 11, 12])


if __name__ == "__main__":
    unittest.main()
