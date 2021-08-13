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

import paddle
from onnxbase import APIOnnx


class Net(paddle.nn.Layer):
    """
    simplr Net
    """

    def __init__(self):
        super(Net, self).__init__()
        self._bn = paddle.nn.Hardsigmoid()

    def forward(self, inputs):
        """
        forward
        """
        x = self._bn(inputs)
        return x


def test_Hardsigmoid_9():
    """
    api: paddle.nn.Hardsigmoid
    op version: 9
    """
    op = Net()
    # func, name, input_shape, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, 'nn_Hardsigmoid', [3, 1, 10, 10], [9])
    obj.run()


def test_Hardsigmoid_10():
    """
    api: paddle.nn.Hardsigmoid
    op version: 10
    """
    op = Net()
    # func, name, input_shape, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, 'nn_Hardsigmoid', [3, 1, 10, 10], [10])
    obj.run()


def test_Hardsigmoid_11():
    """
    api: paddle.nn.Hardsigmoid
    op version: 11
    """
    op = Net()
    # func, name, input_shape, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, 'nn_Hardsigmoid', [3, 1, 10, 10], [11])
    obj.run()


def test_Hardsigmoid_12():
    """
    api: paddle.nn.Hardsigmoid
    op version: 12
    """
    op = Net()
    # func, name, input_shape, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, 'nn_Hardsigmoid', [3, 1, 10, 10], [12])
    obj.run()
