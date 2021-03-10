# Copyright (c) 2020  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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

from __future__ import absolute_import

import numpy as np
import paddle
from paddle.fluid import layers
from paddle2onnx.constant import dtypes
from paddle2onnx.op_mapper import CustomPaddleOp, register_custom_paddle_op


class FillConstantBatchSizeLike(CustomPaddleOp):
    def __init__(self, node, **kw):
        super(FillConstantBatchSizeLike, self).__init__(node)

    def forward(self):
        input = self.input('Input', 0)
        input_shape = paddle.shape(input)
        updates = input_shape[self.node.attr('input_dim_idx')]
        shape = paddle.assign(np.array(self.node.attr('shape')).astype('int32'))
        dims = len(self.node.attr('shape'))
        new_shape = paddle.concat([shape[:self.node.attr('output_dim_idx')], updates, shape[self.node.attr('output_dim_idx')+1:dims]])
        dtype = dtypes.DTYPE_PADDLE_STR_MAP[self.node.attr('dtype')]
        out = paddle.full(new_shape, self.node.attr('value'), dtype)
        return {'Out': [out]}


register_custom_paddle_op('fill_constant_batch_size_like', FillConstantBatchSizeLike)
