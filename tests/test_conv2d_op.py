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

import numpy as np
from op_test import OpTest


class TestConv2dOp(OpTest):
    def setUp(self):
        self.op_type = "conv2d"
        self.use_cudnn = False
        self.use_mkldnn = False
        self.dtype = np.float32
        self.groups = 1
        self.dilations = [1, 1]
        self.pad = [0, 0]
        self.stride = [1, 1]
        self.input_size = [2, 3, 5, 5]
        f_c = self.input_size[1] / self.groups
        self.filter_size = [6, f_c, 3, 3]

        conv2d_param = {
            'stride': self.stride,
            'pad': self.pad,
            'dilation': self.dilations
        }

        input = np.random.random(self.input_size).astype(self.dtype)
        filter = np.random.random(self.filter_size).astype(self.dtype)

        self.inputs = {'Input': input, 'Filter': filter}

        self.persistable = ['Filter']

        self.attrs = {
            'strides': self.stride,
            'paddings': self.pad,
            'groups': self.groups,
            'dilations': self.dilations,
            'use_cudnn': self.use_cudnn,
            'use_mkldnn': self.use_mkldnn
        }

        output = np.zeros((1, 1, 1, 1))
        self.outputs = {'Output': output}

    def test_check_output(self):
        self.check_output(decimal=5)


if __name__ == '__main__':
    unittest.main()
