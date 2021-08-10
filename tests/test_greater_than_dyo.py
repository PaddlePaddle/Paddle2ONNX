#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
paddle.greater_than
"""
import paddle
from onnxbase import APIOnnx


class Net(paddle.nn.Layer):
    """
    simplr Net
    """

    def __init__(self):
        super(Net, self).__init__()

    def forward(self, inputs, inputs_):
        """
        forward
        """
        x = paddle.greater_than(inputs, inputs_)
        return x.astype('float32')


def test_greater_than_9():
    """
    api: paddle.greater_than
    op version: 9
    """
    op = Net()
    op.eval()
    # func, name, input_shape, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, 'greater_than', [3, 10], [9], binputs=True)
    obj.run()


def test_greater_than_10():
    """
    api: paddle.greater_than
    op version: 9
    """
    op = Net()
    op.eval()
    # func, name, input_shape, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, 'greater_than', [3, 10], [10], binputs=True)
    obj.run()


def test_greater_than_11():
    """
    api: paddle.greater_than
    op version: 11
    """
    op = Net()
    op.eval()
    # func, name, input_shape, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, 'greater_than', [3, 10], [11], binputs=True)
    obj.run()


def test_greater_than_12():
    """
    api: paddle.greater_than
    op version: 12
    """
    op = Net()
    op.eval()
    # func, name, input_shape, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, 'greater_than', [3, 10], [12], binputs=True)
    obj.run()
