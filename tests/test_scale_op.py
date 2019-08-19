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


class TestScaleOp(OpTest):
    def setUp(self):
        self.op_type = "scale"
        self.dtype = np.float32
        self.inputs = {'X': np.random.random((10, 10)).astype(self.dtype)}
        self.attrs = {'scale': -2.3}
        self.outputs = {
            'Out': self.inputs['X'] * self.dtype(self.attrs['scale'])
        }

    def test_check_output(self):
        self.check_output()


class TestScale1Op(OpTest):
    def setUp(self):
        self.op_type = "scale"
        self.dtype = np.float32
        self.inputs = {'X': np.random.random((10, 10)).astype(self.dtype)}
        self.attrs = {'scale': -2.3, "bias_after_scale": True}
        self.outputs = {
            'Out': self.inputs['X'] * self.dtype(self.attrs['scale'])
        }

    def test_check_output(self):
        self.check_output()


if __name__ == '__main__':
    unittest.main()
