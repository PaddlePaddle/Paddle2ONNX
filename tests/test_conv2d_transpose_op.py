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


class TestConv2dTransposeOp(OpTest):
    def setUp(self):
        self.use_cudnn = False
        self.init_op_type()
        self.init_test_case()

        input_ = np.random.random(self.input_size).astype('float32')
        filter_ = np.random.random(self.filter_size).astype('float32')

        self.inputs = {'Input': input_, 'Filter': filter_}
        self.attrs = {
            'strides': self.stride,
            'paddings': self.pad,
            'dilations': self.dilations,
            'use_cudnn': self.use_cudnn,
            'data_format': 'AnyLayout'
        }

        self.outputs = {'Output': np.zeros((1, 1))}

    def init_test_case(self):
        self.pad = [0, 0]
        self.stride = [1, 1]
        self.dilations = [1, 1]
        self.input_size = [2, 3, 5, 5]  # NCHW
        f_c = self.input_size[1]
        self.filter_size = [f_c, 6, 3, 3]

    def init_op_type(self):
        self.op_type = 'conv2d_transpose'

    def test_check_output(self):
        self.check_output()


if __name__ == '__main__':
    unittest.main()
