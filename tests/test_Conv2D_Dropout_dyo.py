#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
paddle.nn.Conv2D, paddle.nn.Dropout
"""
import paddle
from onnxbase import APIOnnx


class Net(paddle.nn.Layer):
    """
    simplr Net
    """

    def __init__(self):
        super(Net, self).__init__()
        self._bn = paddle.nn.Conv2D(
            in_channels=1, out_channels=2, kernel_size=3)
        self._drop = paddle.nn.Dropout(p=0.5)

    def forward(self, inputs):
        """
        forward
        """
        x = self._bn(inputs)
        x = self._drop(x)
        return x


def test_Conv2D_Dropout_9():
    """
    api: paddle.nn.Conv2D, paddle.nn.Dropout
    op version: 9
    """
    op = Net()
    op.eval()
    # func, name, input_shape, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, 'Conv2D', [3, 1, 10, 10], [9])
    obj.run()


def test_Conv2D_Dropout_10():
    """
    api: paddle.nn.Conv2D, paddle.nn.Dropout
    op version: 10
    """
    op = Net()
    op.eval()
    # func, name, input_shape, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, 'Conv2D', [3, 1, 10, 10], [10])
    obj.run()


def test_Conv2D_Dropout_11():
    """
    api: paddle.nn.Conv2D, paddle.nn.Dropout
    op version: 11
    """
    op = Net()
    op.eval()
    # func, name, input_shape, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, 'Conv2D', [3, 1, 10, 10], [11])
    obj.run()


def test_Conv2D_Dropout_12():
    """
    api: paddle.nn.Conv2D, paddle.nn.Dropout
    op version: 12
    """
    op = Net()
    op.eval()
    # func, name, input_shape, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, 'Conv2D', [3, 1, 10, 10], [12])
    obj.run()
