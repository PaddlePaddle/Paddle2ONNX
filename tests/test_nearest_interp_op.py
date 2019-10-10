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
from op_test import OpTest
import paddle.fluid.core as core


def nearest_neighbor_interp_np(X,
                               out_h,
                               out_w,
                               out_size=None,
                               actual_shape=None,
                               align_corners=True):
    """nearest neighbor interpolation implement in shape [N, C, H, W]"""
    if out_size is not None:
        out_h = out_size[0]
        out_w = out_size[1]
    if actual_shape is not None:
        out_h = actual_shape[0]
        out_w = actual_shape[1]
    n, c, in_h, in_w = X.shape

    ratio_h = ratio_w = 0.0
    if (out_h > 1):
        if (align_corners):
            ratio_h = (in_h - 1.0) / (out_h - 1.0)
        else:
            ratio_h = 1.0 * in_h / out_h
    if (out_w > 1):
        if (align_corners):
            ratio_w = (in_w - 1.0) / (out_w - 1.0)
        else:
            ratio_w = 1.0 * in_w / out_w

    out = np.zeros((n, c, out_h, out_w))

    if align_corners:
        for i in range(out_h):
            in_i = int(ratio_h * i + 0.5)
            for j in range(out_w):
                in_j = int(ratio_w * j + 0.5)
                out[:, :, i, j] = X[:, :, in_i, in_j]
    else:
        for i in range(out_h):
            in_i = int(ratio_h * i)
            for j in range(out_w):
                in_j = int(ratio_w * j)
                out[:, :, i, j] = X[:, :, in_i, in_j]
    return out.astype(X.dtype)


class TestNearestInterpOp(OpTest):
    def setUp(self):
        self.out_size = None
        self.actual_shape = None
        self.init_test_case()
        self.op_type = "nearest_interp"
        input_np = np.random.random(self.input_shape).astype("float32")

        if self.scale > 0:
            out_h = int(self.input_shape[2] * self.scale)
            out_w = int(self.input_shape[3] * self.scale)
        else:
            out_h = self.out_h
            out_w = self.out_w

        output_np = nearest_neighbor_interp_np(input_np, out_h, out_w,
                                               self.out_size, self.actual_shape,
                                               self.align_corners)
        self.inputs = {'X': input_np}
        if self.out_size is not None:
            self.inputs['OutSize'] = self.out_size
        if self.actual_shape is not None:
            self.inputs['OutSize'] = self.actual_shape
        self.attrs = {
            'out_h': self.out_h,
            'out_w': self.out_w,
            'scale': self.scale,
            'interp_method': self.interp_method,
            'align_corners': self.align_corners,
        }
        self.outputs = {'Out': output_np}

    def test_check_output(self):
        self.check_output(is_interp=True)

#    def test_check_grad(self):
#        self.check_grad(['X'], 'Out', in_place=True)

    def init_test_case(self):
        self.interp_method = 'nearest'
        self.input_shape = [1, 3, 4, 4]
        self.out_h = 16
        self.out_w = 16
        self.scale = 0.
        self.out_size = np.array([16, 16]).astype("int32")
        self.align_corners = False


class TestNearestNeighborInterpCase5(TestNearestInterpOp):
    def init_test_case(self):
        self.interp_method = 'nearest'
        self.input_shape = [1, 3, 9, 6]
        self.out_h = 12
        self.out_w = 12
        self.scale = 0.
        self.out_size = np.array([11, 11]).astype("int32")
        self.align_corners = False


class TestNearestNeighborInterpCase6(TestNearestInterpOp):
    def init_test_case(self):
        self.interp_method = 'nearest'
        self.input_shape = [1, 1, 128, 64]
        self.out_h = 64
        self.out_w = 128
        self.scale = 0.
        self.out_size = np.array([140, 129]).astype("int32")
        self.align_corners = False


class TestNearestNeighborInterpActualShape(TestNearestInterpOp):
    def init_test_case(self):
        self.interp_method = 'nearest'
        self.input_shape = [1, 2, 32, 16]
        self.out_h = 64
        self.out_w = 32
        self.scale = 0.
        self.out_size = np.array([66, 40]).astype("int32")
        self.align_corners = False


if __name__ == "__main__":
    unittest.main()
