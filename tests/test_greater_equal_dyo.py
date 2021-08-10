#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
paddle.greater_equal
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
        x = paddle.greater_equal(inputs, inputs_)
        return x.astype('float32')


def test_greater_equal_12():
    """
    api: paddle.greater_equal
    op version: 12
    """
    op = Net()
    op.eval()
    # func, name, input_shape, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, 'greater_equal', [3, 10], [12], binputs=True)
    obj.run()
