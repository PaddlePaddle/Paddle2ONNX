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
from paddle2onnx.op_mapper import CustomPaddleOp, register_custom_paddle_op


class DeformConv2d(CustomPaddleOp):
    dtype_mapping = {
        "VarType.INT32": "int32",
        "VarType.INT64": "int64",
        "VarType.FP32": "float32",
        "VarType.FP64": "float64",
    }

    def __init__(self, node, **kw):
        super(DeformConv2d, self).__init__(node)
        self.x_shape = node.input_shape('Input', 0)
        self.offset_shape = node.input_shape('Offset', 0)
        self.filter_shape = node.input_shape('Filter', 0)
        self.stride = node.attr('strides')[0]
        self.padding = node.attr('paddings') + node.attr('paddings')
        self.groups = node.attr('groups')
        self.dilation = node.attr('dilations')[0]

        self.kernel_size = self.filter_shape[2]
        self.N = self.kernel_size**2
        self.offset_h = self.offset_shape[2]
        self.offset_w = self.offset_shape[3]
        self.num_filters = self.filter_shape[0]
        self.padded_x_w = self.x_shape[3] + self.padding[0] * 2
        self.padded_x_h = self.x_shape[2] + self.padding[1] * 2

    def forward(self):
        input = self.input('Input', 0)
        weight = self.input('Filter', 0)
        mask = self.input('Mask', 0)
        offset = self.input('Offset', 0)
        input = layers.pad2d(input, self.padding)
        offset_x = paddle.strided_slice(
            offset,
            axes=[1],
            starts=[0],
            ends=[self.offset_shape[1]],
            strides=[2])
        offset_y = paddle.strided_slice(
            offset,
            axes=[1],
            starts=[1],
            ends=[self.offset_shape[1]],
            strides=[2])

        offset = paddle.concat([offset_x, offset_y], axis=1)
        offset_shape = paddle.shape(offset)

        # (b, 2self.N, h, w)
        p = self._get_p(offset, 'float32', offset_shape)

        # (b, h, w, 2self.N)
        p = p.transpose((0, 2, 3, 1))
        q_lt = p.floor()
        q_rb = q_lt + 1

        padded_x_shape = paddle.shape(input)
        padded_w = paddle.cast(padded_x_shape[3], dtype='float32')
        padded_h = paddle.cast(padded_x_shape[2], dtype='float32')

        q_lt = paddle.cast(
            paddle.concat(
                [
                    paddle.clip(q_lt[:, :, :, :self.N], 0, padded_w - 1),
                    paddle.clip(q_lt[:, :, :, self.N:], 0, padded_h - 1)
                ],
                axis=-1),
            dtype='int64')
        q_rb = paddle.cast(
            paddle.concat(
                [
                    paddle.clip(q_rb[:, :, :, :self.N], 0, padded_w - 1),
                    paddle.clip(q_rb[:, :, :, self.N:], 0, padded_h - 1)
                ],
                axis=-1),
            dtype='int64')
        q_lb = paddle.concat(
            [q_lt[:, :, :, :self.N], q_rb[:, :, :, self.N:]], axis=-1)
        q_rt = paddle.concat(
            [q_rb[:, :, :, :self.N], q_lt[:, :, :, self.N:]], axis=-1)

        # clip p
        p = paddle.concat(
            [
                paddle.clip(p[:, :, :, :self.N], 0, padded_w - 1),
                paddle.clip(p[:, :, :, self.N:], 0, padded_h - 1)
            ],
            axis=-1)

        # bilinear kernel (b, h, w, self.N)
        g_lt = (
            1 + (paddle.cast(
                q_lt[:, :, :, :self.N], dtype='float32') - p[:, :, :, :self.N])
        ) * (1 + paddle.cast(
            q_lt[:, :, :, self.N:], dtype='float32') - p[:, :, :, self.N:])
        g_rb = (
            1 - (paddle.cast(
                q_rb[:, :, :, :self.N], dtype='float32') - p[:, :, :, :self.N])
        ) * (1 - (paddle.cast(
            q_rb[:, :, :, self.N:], dtype='float32') - p[:, :, :, self.N:]))
        g_lb = (
            1 + (paddle.cast(
                q_lb[:, :, :, :self.N], dtype='float32') - p[:, :, :, :self.N])
        ) * (1 - (paddle.cast(
            q_lb[:, :, :, self.N:], dtype='float32') - p[:, :, :, self.N:]))
        g_rt = (
            1 - (paddle.cast(
                q_rt[:, :, :, :self.N], dtype='float32') - p[:, :, :, :self.N])
        ) * (1 + paddle.cast(
            q_rt[:, :, :, self.N:], dtype='float32') - p[:, :, :, self.N:])
        # (b, c, h, w, self.N)

        x_q_lt = self._get_x_q(input, q_lt, offset_shape)
        x_q_rb = self._get_x_q(input, q_rb, offset_shape)
        x_q_lb = self._get_x_q(input, q_lb, offset_shape)
        x_q_rt = self._get_x_q(input, q_rt, offset_shape)

        # (b, c, h, w, self.N)
        x_offset = paddle.unsqueeze(g_lt, 1) * x_q_lt + \
                   paddle.unsqueeze(g_rb, 1) * x_q_rb + \
                   paddle.unsqueeze(g_lb, 1) * x_q_lb + \
                   paddle.unsqueeze(g_rt, 1) * x_q_rt

        # modulation
        if mask is not None:
            mask = paddle.transpose(mask, (0, 2, 3, 1))
            mask = paddle.unsqueeze(mask, 1)
            mask = paddle.tile(mask, [1, self.x_shape[1], 1, 1, 1])
            x_offset *= mask

        x_offset = self._reshape_x_offset(x_offset)

        out = paddle.nn.functional.conv2d(
            x_offset, weight, stride=self.kernel_size, groups=self.groups)

        return {'Output': [out]}

    def _get_p_n(self, dtype):
        p_n_x = paddle.arange(
            0,
            self.kernel_size + (self.kernel_size - 1) * (self.dilation - 1),
            step=self.dilation,
            dtype=dtype)
        p_n_x = p_n_x.unsqueeze(1)
        p_n_x = paddle.tile(p_n_x, [1, self.kernel_size])
        p_n_y = paddle.arange(
            0,
            self.kernel_size + (self.kernel_size - 1) * (self.dilation - 1),
            step=self.dilation,
            dtype=dtype)
        p_n_y = p_n_y.unsqueeze(0)
        p_n_y = paddle.tile(p_n_y, [self.kernel_size, 1])

        # (2N, 1)
        p_n_x = paddle.reshape(p_n_x, [-1])
        p_n_y = paddle.reshape(p_n_y, [-1])
        p_n = paddle.concat([p_n_x, p_n_y], -1)
        p_n = paddle.reshape(p_n, (1, 2 * self.N, 1, 1))

        return p_n

    def _get_p_0(self, dtype, offset_shape):
        p_0_x = paddle.arange(
            0, offset_shape[3] * self.stride, step=self.stride, dtype=dtype)
        p_0_x = p_0_x.unsqueeze(1)
        p_0_x = paddle.expand(p_0_x, offset_shape[2:])

        p_0_y = paddle.arange(
            0, offset_shape[3] * self.stride, step=self.stride, dtype=dtype)
        p_0_y = p_0_y.unsqueeze(0)
        p_0_y = paddle.expand(p_0_y, offset_shape[2:])

        p_0_x = p_0_x.unsqueeze([0, 1])
        p_0_x = paddle.tile(p_0_x, (1, self.N, 1, 1))

        p_0_y = p_0_y.unsqueeze([0, 1])
        p_0_y = paddle.tile(p_0_y, (1, self.N, 1, 1))

        p_0 = paddle.concat([p_0_x, p_0_y], 1)

        return p_0

    def _get_p(self, offset, dtype, offset_shape):
        # (1, 2N, 1, 1)
        p_n = self._get_p_n(dtype)

        # (1, 2N, h, w)
        p_0 = self._get_p_0(dtype, offset_shape)
        p = offset + p_0 + p_n

        return p

    def _get_x_q(self, x, q, offset_shape):
        c = self.x_shape[1]
        # (b, c, h*w)
        x = paddle.reshape(x, [0, 0, -1])

        # (b, h, w, self.N)
        index = paddle.cast(
            q[:, :, :, :self.N] * self.padded_x_w,
            dtype='int64') + q[:, :, :, self.N:]  # offset_x*w + offset_y

        # (b, c, h*w*self.N)
        index = paddle.unsqueeze(index, 1)
        index = paddle.tile(index, [1, c, 1, 1, 1])
        index = paddle.reshape(index, (0, 0, -1))

        x_range = list(range(3))
        dim = 2
        x_range[0] = dim
        x_range[dim] = 0
        x_swaped = paddle.transpose(x, perm=x_range)
        index_range = list(range(3))
        index_range[0] = dim
        index_range[dim] = 0
        index_swaped = paddle.transpose(index, perm=index_range)
        x_shape = layers.shape(x_swaped)
        index_shape = layers.shape(index_swaped)

        prod = paddle.prod(x_shape[1:], keepdim=True)
        x_swaped_flattend = paddle.reshape(x_swaped, [-1])

        index_swaped_flattend = paddle.reshape(index_swaped, [-1])
        index_swaped_flattend *= prod

        bias = paddle.arange(start=0, end=prod, step=1, dtype='float32')
        bias = paddle.tile(bias, index_shape[0])

        index_swaped_flattend += bias

        gathered = paddle.gather(x_swaped_flattend, index_swaped_flattend)
        gathered = paddle.reshape(gathered, layers.shape(index_swaped))

        x_offset = paddle.transpose(gathered, perm=x_range)
        x_offset = paddle.reshape(x_offset,
                                  (-1, c, self.offset_h, self.offset_w, self.N))

        return x_offset

    def _reshape_x_offset(self, x_offset):
        c = self.x_shape[1]
        x_offset = paddle.concat(
            [
                paddle.reshape(x_offset[:, :, :, :, s:s + self.kernel_size], (
                    -1, c, self.offset_h, self.offset_w * self.kernel_size))
                for s in range(0, self.N, self.kernel_size)
            ],
            axis=-1)
        x_offset = paddle.reshape(x_offset,
                                  (-1, c, self.offset_h * self.kernel_size,
                                   self.offset_w * self.kernel_size))
        return x_offset


register_custom_paddle_op('deformable_conv', DeformConv2d)
