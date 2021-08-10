#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
paddle.nn.GroupNorm
"""
import paddle
from onnxbase import APIOnnx


class Net(paddle.nn.Layer):
    """
    simplr Net
    """

    def __init__(self):
        super(Net, self).__init__()
        self._bn = paddle.nn.GroupNorm(num_groups=5, num_channels=10)

    def forward(self, inputs):
        """
        forward
        """
        x = self._bn(inputs)
        return x


def test_GroupNorm_11():
    """
    api: paddle.nn.GroupNorm
    op version: 11
    """
    op = Net()
    # func, name, input_shape, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, 'nn_GroupNorm', [5, 10, 8, 8], [11])
    obj.run()


def test_GroupNorm_12():
    """
    api: paddle.nn.GroupNorm
    op version: 12
    """
    op = Net()
    # func, name, input_shape, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, 'nn_GroupNorm', [5, 10, 8, 8], [12])
    obj.run()
