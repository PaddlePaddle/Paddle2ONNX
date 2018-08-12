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


class TestReduceSumOp(OpTest):
    def setUp(self):
        self.init_op_type()
        self.init_keep_dim()
        self.init_reduce_all()
        self.inputs = {'X': np.random.random((5, 6, 7, 8)).astype('float32')}
        self.attrs = {'dim': [2], }
        self.outputs = {'Out': np.zeros((1, 1))}

    def init_op_type(self):
        self.op_type = 'reduce_sum'

    def init_keep_dim(self):
        self.keep_dim = True

    def init_reduce_all(self):
        self.reduce_all = False

    def test_check_output(self):
        self.check_output(decimal=4)


class TestReduceMeanOp(TestReduceSumOp):
    def init_op_type(self):
        self.op_type = 'reduce_mean'

    def init_reduce_all(self):
        self.reduce_all = True


class TestReduceMaxOp(TestReduceSumOp):
    def init_op_type(self):
        self.op_type = 'reduce_max'

    def init_keep_dim(self):
        self.keep_dim = False

    def init_reduce_all(self):
        self.reduce_all = True


class TestReduceMinOp(TestReduceSumOp):
    def init_op_type(self):
        self.op_type = 'reduce_min'

    def init_keep_dim(self):
        self.keep_dim = False


if __name__ == '__main__':
    unittest.main()
