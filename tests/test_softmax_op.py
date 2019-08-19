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

import unittest
import numpy as np
from op_test import OpTest


class TestSoftMaxOP(OpTest):
    def setUp(self):
        self.op_type = "softmax"
        self.init_dtype()

        x = np.random.uniform(-1, 1, [10, 12]).astype(self.dtype)
        max_x = np.max(x, axis=1).reshape((-1, 1))
        exp_x = np.exp(x - max_x)
        out = exp_x / np.sum(exp_x, axis=1).reshape((-1, 1))

        self.inputs = {'X': x}
        self.outputs = {'Out': out}

    def init_dtype(self):
        self.dtype = np.float32

    def test_check_output(self):
        self.check_output()


class TestSoftMaxOP1(OpTest):
    def setUp(self):
        self.op_type = "softmax"
        self.init_dtype()

        x = np.random.uniform(-1, 1, [10, 11, 12]).astype(self.dtype)
        out = x

        self.inputs = {'X': x}
        self.outputs = {'Out': out}

    def init_dtype(self):
        self.dtype = np.float32

    def test_check_output(self):
        self.check_output()


if __name__ == '__main__':
    unittest.main()
