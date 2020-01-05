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


@unittest.skip("ONNX FATAL: Don't know how to translate op SequenceConstruct and ConcatFromSequence")
class TestStackOp(OpTest):
    def setUp(self):
        self.op_type = 'stack'
        self.init_test_data()
        self.inputs = {'X': [('x0', self.x0), ('x1', self.x1), ('x2', self.x2)]}
        self.attrs = {'axis': self.axis}
        self.outputs = {'Out': np.zeros((1, 1)).astype('float32')}

    def test_check_output(self):
        self.check_output()

    def init_test_data(self):
        self.x0 = np.random.random((1, 2, 3)).astype('float32')
        self.x1 = np.random.random((1, 2, 3)).astype('float32')
        self.x2 = np.random.random((1, 2, 3)).astype('float32')
        self.axis = 0


@unittest.skip("ONNX FATAL: Don't know how to translate op SequenceConstruct and ConcatFromSequence")
class TestStackOp2(TestStackOp):
    def init_test_data(self):
        self.x0 = np.random.random((1, 2, 3)).astype('float32')
        self.x1 = np.random.random((1, 2, 3)).astype('float32')
        self.x2 = np.random.random((1, 2, 3)).astype('float32')
        self.axis = 1


if __name__ == '__main__':
    unittest.main()
