#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
paddle.nn.Hardswish
"""
import paddle
from onnxbase import APIOnnx


class Net(paddle.nn.Layer):
    """
    simplr Net
    """

    def __init__(self):
        super(Net, self).__init__()
        self._bn = paddle.nn.Hardswish()

    def forward(self, inputs):
        """
        forward
        """
        x = self._bn(inputs)
        return x


def test_Hardswish_9():
    """
    api: paddle.nn.Hardswish
    op version: 9
    """
    op = Net()
    # func, name, input_shape, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, 'nn_Hardswish', [3, 1, 10, 10], [9])
    obj.run()


def test_Hardswish_10():
    """
    api: paddle.nn.Hardswish
    op version: 10
    """
    op = Net()
    # func, name, input_shape, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, 'nn_Hardswish', [3, 1, 10, 10], [10])
    obj.run()


def test_Hardswish_11():
    """
    api: paddle.nn.Hardswish
    op version: 11
    """
    op = Net()
    # func, name, input_shape, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, 'nn_Hardswish', [3, 1, 10, 10], [11])
    obj.run()


def test_Hardswish_12():
    """
    api: paddle.nn.Hardswish
    op version: 12
    """
    op = Net()
    # func, name, input_shape, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, 'nn_Hardswish', [3, 1, 10, 10], [12])
    obj.run()
