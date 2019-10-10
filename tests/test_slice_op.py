#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
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

from __future__ import print_function

import unittest
import numpy as np
import paddle.fluid.core as core
from op_test import OpTest


class TestSliceOp(OpTest):
    def setUp(self):
        self.op_type = "slice"
        self.config()
        self.inputs = {'Input': self.input}
        self.outputs = {'Out': self.out}
        self.attrs = {
            'axes': self.axes,
            'starts': self.starts,
            'ends': self.ends
        }

    def config(self):
        self.input = np.random.random([20, 10, 5]).astype("float32")
        self.starts = [0, 0, 3]
        self.ends = [20, 10, 4]
        self.axes = [0, 1, 2]
        self.out = self.input[:, :, 3:4]

    def test_check_output(self):
        self.check_output(is_onnxruntime=True)


class TestSliceOp_decs_dim(OpTest):
    def setUp(self):
        self.op_type = "slice"
        self.config()
        self.inputs = {'Input': self.input}
        self.outputs = {'Out': self.out}
        self.attrs = {
            'axes': self.axes,
            'starts': self.starts,
            'ends': self.ends
        }

    def config(self):
        self.input = np.random.random([3, 4, 5, 6]).astype("float32")
        self.starts = [1, 0, 2]
        self.ends = [2, 3, 4]
        self.axes = [0, 1, 2]
        self.decrease_axis = [0]
        self.out = self.input[1, 0:3, 2:4, :]

    def test_check_output(self):
        self.check_output(is_onnxruntime=True)


class TestSliceOp_decs_dim_2(OpTest):
    def setUp(self):
        self.op_type = "slice"
        self.config()
        self.inputs = {'Input': self.input}
        self.outputs = {'Out': self.out}
        self.attrs = {
            'axes': self.axes,
            'starts': self.starts,
            'ends': self.ends
        }

    def config(self):
        self.input = np.random.random([3, 4, 5, 6]).astype("float32")
        self.starts = [1, 0, 2]
        self.ends = [2, 1, 4]
        self.axes = [0, 1, 2]
        self.decrease_axis = [0, 1]
        self.out = self.input[1, 0, 2:4, :]

    def test_check_output(self):
        self.check_output(is_onnxruntime=True)


class TestSliceOp_decs_dim_3(OpTest):
    def setUp(self):
        self.op_type = "slice"
        self.config()
        self.inputs = {'Input': self.input}
        self.outputs = {'Out': self.out}
        self.attrs = {
            'axes': self.axes,
            'starts': self.starts,
            'ends': self.ends
        }

    def config(self):
        self.input = np.random.random([3, 4, 5, 6]).astype("float32")
        self.starts = [-1, 0, 2]
        self.ends = [1000000, 1, 4]
        self.axes = [0, 1, 2]
        self.decrease_axis = [0, 1]
        self.out = self.input[-1, 0, 2:4, :]

    def test_check_output(self):
        self.check_output(is_onnxruntime=True)


if __name__ == '__main__':
    unittest.main()
