#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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


def generate_compatible_shapes(dim_X, dim_Y, transpose_X, transpose_Y):
    BATCH_SIZE = 2
    M = 3
    N = 4
    K = 5
    assert dim_X >= 1
    assert dim_Y >= 1
    shape_X, shape_Y = [], []  # suppress IDE warning
    if (dim_X == 1 and transpose_X) or (dim_Y == 1 and transpose_Y):
        K = 1
    if dim_X == 1:
        if transpose_X:
            shape_X = [M]
        else:
            shape_X = [K]
    if dim_Y == 1:
        if transpose_Y:
            shape_Y = [N]
        else:
            shape_Y = [K]
    if dim_X >= 2:
        if transpose_X:
            shape_X = [K, M]
        else:
            shape_X = [M, K]
    if dim_X == 3:
        shape_X = [BATCH_SIZE] + shape_X
    if dim_Y >= 2:
        if transpose_Y:
            shape_Y = [N, K]
        else:
            shape_Y = [K, N]
    if dim_Y == 3:
        shape_Y = [BATCH_SIZE] + shape_Y
    return shape_X, shape_Y


def reference_matmul(X, Y, transpose_X=False, transpose_Y=False):
    """Reference forward implementation using np.matmul."""
    # np.matmul does not support the transpose flags, so we manually
    # transpose X and Y appropriately.
    if transpose_X:
        X = transpose(X)
    if transpose_Y:
        Y = transpose(Y)

    Out = np.matmul(X, Y)
    if not Out.shape:
        # We do not support 0-dimensional Tensors (scalars). So where
        # np.matmul outputs a scalar, we must convert to a Tensor of
        # shape (1, ) instead.
        # Everywhere else, we are compatible with np.matmul.
        Out = np.array([Out], dtype="float32")
    return Out


def transpose(X):
    if X.ndim == 1:
        X = X.reshape((X.size, 1))
    elif X.ndim == 2:
        X = X.T
    else:
        dim = [i for i in range(len(X.shape))]
        dim[-1], dim[len(X.shape) - 2] = dim[len(X.shape) - 2], dim[-1]
        X = np.transpose(X, tuple(dim))
    return X


class TestMatMulRank2NoTransposeOp(OpTest):
    def setUp(self):
        self.op_type = "matmul"
        self.alpha = 1.0  # default
        self.set_shapes()
        X = np.random.random(self.shape_X).astype("float32")
        Y = np.random.random(self.shape_Y).astype("float32")
        Out = reference_matmul(X, Y, self.transpose_X, self.transpose_Y)
        self.inputs = {'X': X, 'Y': Y}
        self.attrs = {
            'transpose_X': self.transpose_X,
            'transpose_Y': self.transpose_Y,
            'alpha': self.alpha
        }
        self.outputs = {'Out': Out}

    def set_shapes(self):
        self.transpose_X = False
        self.transpose_Y = False
        self.shape_X, self.shape_Y = generate_compatible_shapes(2, 2, self.transpose_X, self.transpose_Y)

    def test_check_output(self):
        self.check_output()


class TestMatMulRank3NoTransposeOp(TestMatMulRank2NoTransposeOp):
    def set_shapes(self):
        self.transpose_X = False
        self.transpose_Y = False
        self.shape_X, self.shape_Y = generate_compatible_shapes(3, 3, self.transpose_X, self.transpose_Y)

    def test_check_output(self):
        self.check_output()


class TestMatMulRank1NoTransposeOp(TestMatMulRank2NoTransposeOp):
    def set_shapes(self):
        self.transpose_X = False
        self.transpose_Y = False
        self.shape_X, self.shape_Y = [3], [3]

    def test_check_output(self):
        self.check_output()


class TestMatMulRank2TransposeYOp(TestMatMulRank2NoTransposeOp):
    def set_shapes(self):
        self.transpose_X = False
        self.transpose_Y = True
        self.shape_X, self.shape_Y = generate_compatible_shapes(2, 2, self.transpose_X, self.transpose_Y)

    def test_check_output(self):
        self.check_output()


class TestMatMulRank5NoTransposeYOp(TestMatMulRank2NoTransposeOp):
    def set_shapes(self):
        """
        Example from
          http://christopher5106.github.io/deep/learning/2018/10/28/understand-batch-matrix-multiplication.html
          but with transpose
        """
        self.transpose_X = False
        self.transpose_Y = True
        self.shape_X, self.shape_Y = (9, 8, 7, 4, 2), (9, 8, 7, 5, 2)

    def test_check_output(self):
        self.check_output()


class TestMatMulShape_1_512_1_TransposeYOp(TestMatMulRank2NoTransposeOp):
    def set_shapes(self):
        """
        Example from ERNIE
        """
        self.transpose_X = False
        self.transpose_Y = True
        self.shape_X, self.shape_Y = (1, 512, 1), (1, 512, 1)

    def test_check_output(self):
        self.check_output()


class TestMatMulRank3Alpha2Op(TestMatMulRank2NoTransposeOp):
    def set_shapes(self):
        self.transpose_X = True
        self.transpose_Y = True
        self.shape_X, self.shape_Y = generate_compatible_shapes(3, 3, self.transpose_X, self.transpose_Y)
        self.alpha = 2.0

    def test_check_output(self):
        self.check_output()


class TestMatMulRank1Alpha2Op(TestMatMulRank1NoTransposeOp):
    def set_shapes(self):
        super().set_shapes()
        self.alpha = 2.0

    def test_check_output(self):
        self.check_output()


if __name__ == "__main__":
    unittest.main()
