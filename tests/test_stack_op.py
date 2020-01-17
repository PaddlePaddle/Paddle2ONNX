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


class TestStackOp(OpTest):
    def setUp(self):
        self.op_type = 'stack'
        self.x0 = np.ones((1, 2, 3), dtype=np.int32)
        self.x1 = np.ones((1, 2, 3), dtype=np.int32)
        self.x2 = np.ones((1, 2, 3), dtype=np.int32)
        self.inputs = {'X': [('x0', self.x0), ('x1', self.x1), ('x2', self.x2)]}
        self.attrs = {'axis': 0}
        self.outputs = {'Y': np.zeros((3, 1, 2, 3), dtype=np.int32)}

    def test_check_output(self):
        self.check_output()


class TestStackOp2(TestStackOp):
    def setUp(self):
        super().setUp()
        self.attrs = {'axis': 1}
        self.outputs = {'Y': np.zeros((1, 3, 2, 3), dtype=np.int32)}

    def test_check_output(self):
        self.check_output()


if __name__ == '__main__':
    unittest.main()
