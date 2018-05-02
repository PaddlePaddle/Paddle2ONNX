#  Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest
import numpy as np
from op_test import OpTest


class TestElementwiseAddOp(OpTest):
    def setUp(self):
        self.attrs = {'axis': 2}
        self.init()

        self.inputs = {
            'X': np.random.random((2, 3, 4, 5)).astype(np.float32),
            'Y': np.random.random((4, 5)).astype(np.float32)
        }

        self.outputs = {'Out': np.zeros((1, 1))}

    def init(self):
        self.op_type = 'elementwise_add'

    def test_check_output(self):
        self.check_output()


class TestElementwiseAddOpNegAxis(OpTest):
    def init(self):
        self.op_type = 'elementwise_add'
        self.attrs = {'axis': -1}


class TestElementwiseSubOp(TestElementwiseAddOp):
    def init(self):
        self.op_type = 'elementwise_sub'


class TestElementwiseMulOp(TestElementwiseAddOp):
    def init(self):
        self.op_type = 'elementwise_mul'


class TestElementwiseDivOp(TestElementwiseAddOp):
    def init(self):
        self.op_type = 'elementwise_div'


class TestElementwisePowOp(TestElementwiseAddOp):
    def init(self):
        self.op_type = 'elementwise_pow'


if __name__ == '__main__':
    unittest.main()
