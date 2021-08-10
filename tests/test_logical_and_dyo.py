#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
paddle.logical_and
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
        x = paddle.logical_and(inputs, inputs_)
        return x.astype('float32')


# def test_logical_and_9():
#     """
#     api: paddle.logical_and
#     op version: 9
#     """
#     op = Net()
#     op.eval()
#     # func, name, input_shape, ver_list, delta=1e-6, rtol=1e-5
#     obj = APIOnnx(op, 'logical_and', [3, 10], [9], binputs=True, data_type='bool')
#     obj.run()


def test_logical_and_10():
    """
    api: paddle.logical_and
    op version: 10
    """
    op = Net()
    op.eval()
    # func, name, input_shape, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(
        op, 'logical_and', [3, 10], [10], binputs=True, data_type='bool')
    obj.run()


def test_logical_and_11():
    """
    api: paddle.logical_and
    op version: 11
    """
    op = Net()
    op.eval()
    # func, name, input_shape, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(
        op, 'logical_and', [3, 10], [11], binputs=True, data_type='bool')
    obj.run()


def test_logical_and_12():
    """
    api: paddle.logical_and
    op version: 12
    """
    op = Net()
    op.eval()
    # func, name, input_shape, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(
        op, 'logical_and', [3, 10], [12], binputs=True, data_type='bool')
    obj.run()
