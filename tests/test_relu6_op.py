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


class TestRelu6Op(OpTest):
    def setUp(self):
        self.op_type = "relu6"
        self.init_dtype()
        x = np.random.uniform(-10, 10, [10, 14]).astype(self.dtype)
        threshold = 6.0
        # The same with TestAbs
        #x[np.abs(x) < 0.005] = 0.02
        #x[np.abs(x - threshold) < 0.005] = threshold + 0.02
        out = np.minimum(np.maximum(x, 0.0), threshold)
        self.attrs = {'threshold': threshold}
        self.inputs = {'X': x}
        self.outputs = {'Out': out}

    def init_dtype(self):
        self.dtype = np.float32

    def test_check_output(self):
        self.check_output()


if __name__ == '__main__':
    unittest.main()
