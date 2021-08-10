#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
paddle.nn.BatchNorm2D
"""
import paddle
from onnxbase import APIOnnx


class Net(paddle.nn.Layer):
    """
    simplr Net
    """

    def __init__(self):
        super(Net, self).__init__()
        self._bn = paddle.nn.BatchNorm2D(num_features=1)

    def forward(self, inputs):
        """
        forward
        """
        x = self._bn(inputs)
        return x


def test_BatchNorm2D_9():
    """
    api: paddle.nn.BatchNorm2D
    op version: 9
    """
    op = Net()
    op.eval()
    # func, name, input_shape, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, 'nn_BatchNorm2D', [3, 1, 10, 10], [9])
    obj.run()


def test_BatchNorm2D_10():
    """
    api: paddle.nn.BatchNorm2D
    op version: 10
    """
    op = Net()
    op.eval()
    # func, name, input_shape, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, 'nn_BatchNorm2D', [3, 1, 10, 10], [10])
    obj.run()


def test_BatchNorm2D_11():
    """
    api: paddle.nn.BatchNorm2D
    op version: 11
    """
    op = Net()
    op.eval()
    # func, name, input_shape, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, 'nn_BatchNorm2D', [3, 1, 10, 10], [11])
    obj.run()


def test_BatchNorm2D_12():
    """
    api: paddle.nn.BatchNorm2D
    op version: 12
    """
    op = Net()
    op.eval()
    # func, name, input_shape, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, 'nn_BatchNorm2D', [3, 1, 10, 10], [12])
    obj.run()
