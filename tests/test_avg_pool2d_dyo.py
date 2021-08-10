#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
paddle.nn.functional.avg_pool2d
"""
import paddle
from onnxbase import APIOnnx


class Net(paddle.nn.Layer):
    """
    simplr Net
    """

    def __init__(self):
        super(Net, self).__init__()

    def forward(self, inputs):
        """
        forward
        """
        x = paddle.nn.functional.avg_pool2d(inputs, kernel_size=2)
        return x


def test_avg_pool2d_9():
    """
    api: paddle.nn.functional.avg_pool2d
    op version: 9
    """
    op = Net()
    op.eval()
    # func, name, input_shape, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, 'avg_pool2d', [3, 1, 10, 10], [9])
    obj.run()


def test_avg_pool2d_10():
    """
    api: paddle.nn.functional.avg_pool2d
    op version: 10
    """
    op = Net()
    op.eval()
    # func, name, input_shape, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, 'avg_pool2d', [3, 1, 10, 10], [10])
    obj.run()


def test_avg_pool2d_11():
    """
    api: paddle.nn.functional.avg_pool2d
    op version: 11
    """
    op = Net()
    op.eval()
    # func, name, input_shape, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, 'avg_pool2d', [3, 1, 10, 10], [11])
    obj.run()


def test_avg_pool2d_12():
    """
    api: paddle.nn.functional.avg_pool2d
    op version: 12
    """
    op = Net()
    op.eval()
    # func, name, input_shape, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, 'avg_pool2d', [3, 1, 10, 10], [12])
    obj.run()
