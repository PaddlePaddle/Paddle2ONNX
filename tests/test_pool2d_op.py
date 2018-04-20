# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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


class TestPool2dOp(OpTest):
    def setUp(self):
        self.op_type = "pool2d"
        self.use_cudnn = False
        self.use_mkldnn = False
        self.dtype = np.float32
        self.init_test_case()
        self.init_global_pool()
        self.init_kernel_type()
        self.init_pool_type()
        self.init_ceil_mode()
        if self.global_pool:
            self.paddings = [0 for _ in range(len(self.paddings))]
        input = np.random.random(self.shape).astype(self.dtype)
        output = np.zeros((1, 1)).astype(self.dtype)

        self.inputs = {'X': input}

        self.attrs = {
            'strides': self.strides,
            'paddings': self.paddings,
            'ksize': self.ksize,
            'pooling_type': self.pool_type,
            'global_pooling': self.global_pool,
            'use_cudnn': self.use_cudnn,
            'use_mkldnn': self.use_mkldnn,
            'ceil_mode': self.ceil_mode,
            'data_format': 'AnyLayout'
        }

        self.outputs = {'Out': output}

    def test_check_output(self):
        self.check_output()

    def init_test_case(self):
        self.shape = [2, 3, 5, 5]
        self.ksize = [3, 3]
        self.strides = [1, 1]
        self.paddings = [0, 0]

    def init_kernel_type(self):
        pass

    def init_pool_type(self):
        self.pool_type = "avg"

    def init_global_pool(self):
        self.global_pool = True

    def init_ceil_mode(self):
        self.ceil_mode = False


class TestPool2dOp1(TestPool2dOp):
    def init_test_case(self):
        self.shape = [2, 3, 7, 7]
        self.ksize = [3, 3]
        self.strides = [1, 1]
        self.paddings = [0, 0]

    def init_pool_type(self):
        self.pool_type = "avg"

    def init_global_pool(self):
        self.global_pool = False


class TestPool2dOp2(TestPool2dOp):
    def init_test_case(self):
        self.shape = [2, 3, 7, 7]
        self.ksize = [3, 3]
        self.strides = [1, 1]
        self.paddings = [1, 1]

    def init_pool_type(self):
        self.pool_type = "avg"

    def init_global_pool(self):
        self.global_pool = False


class TestPool2dOp3(TestPool2dOp):
    def init_pool_type(self):
        self.pool_type = "max"


class TestPool2dOp4(TestPool2dOp1):
    def init_pool_type(self):
        self.pool_type = "max"


class TestPool2dOp5(TestPool2dOp2):
    def init_pool_type(self):
        self.pool_type = "max"


if __name__ == '__main__':
    unittest.main()
