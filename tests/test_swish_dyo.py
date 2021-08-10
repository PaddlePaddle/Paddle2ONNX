#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
paddle.nn.functional.swish
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
        x = paddle.nn.functional.swish(inputs)
        return x


def test_swish_9():
    """
    api: paddle.nn.functional.swish
    op version: 9
    """
    op = Net()
    op.eval()
    # func, name, input_shape, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, 'swish', [3, 10], [9])
    obj.run()


def test_swish_10():
    """
    api: paddle.nn.functional.swish
    op version: 10
    """
    op = Net()
    op.eval()
    # func, name, input_shape, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, 'swish', [3, 10], [10])
    obj.run()


def test_swish_11():
    """
    api: paddle.nn.functional.swish
    op version: 11
    """
    op = Net()
    op.eval()
    # func, name, input_shape, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, 'swish', [3, 10], [11])
    obj.run()


def test_swish_12():
    """
    api: paddle.nn.functional.swish
    op version: 12
    """
    op = Net()
    op.eval()
    # func, name, input_shape, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, 'swish', [3, 10], [12])
    obj.run()
