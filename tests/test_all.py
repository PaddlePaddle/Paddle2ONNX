# Copyright (c) 2021  PaddlePaddle Authors. All Rights Reserved.
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
from onnxbase import randtool


class Net(paddle.nn.Layer):
    """
    simple Net
    """

    def __init__(self, axis=None, keepdim=False):
        super(Net, self).__init__()
        self.axis = axis
        self.keepdim = keepdim

    def forward(self, inputs):
        """
        forward
        """
        x = paddle.all(inputs, axis=self.axis, keepdim=self.keepdim)
        return x


def test_all_10():
    """
    api: paddle.all
    op version: 10
    """
    op = Net()
    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, 'all', [10])
    obj.set_input_data(
        "input_data",
        paddle.to_tensor(randtool("float", -1, 1, [3, 10]).astype('bool')))
    obj.run()


def test_all_11():
    """
    api: paddle.all
    op version: 11
    """
    op = Net()
    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, 'all', [11])
    obj.set_input_data(
        "input_data",
        paddle.to_tensor(randtool("float", -1, 1, [3, 10]).astype('bool')))
    obj.run()


def test_all_12():
    """
    api: paddle.all
    op version: 12
    """
    op = Net()
    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, 'all', [12])
    obj.set_input_data(
        "input_data",
        paddle.to_tensor(randtool("float", -1, 1, [3, 10]).astype('bool')))
    obj.run()


def test_all_keepdim():
    """
    api: paddle.all
    op version: 12
    """
    op = Net(keepdim=True)
    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, 'all', [12])
    obj.set_input_data(
        "input_data",
        paddle.to_tensor(randtool("float", -1, 1, [4, 3, 10]).astype('bool')))
    obj.run()


def test_all_axis():
    """
    api: paddle.all
    op version: 12
    """
    op = Net(axis=1)
    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, 'all', [12])
    obj.set_input_data(
        "input_data",
        paddle.to_tensor(randtool("float", -1, 1, [4, 3, 10]).astype('bool')))
    obj.run()


def test_all_axis_keepdim():
    """
    api: paddle.all
    op version: 12
    """
    op = Net(axis=1, keepdim=True)
    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, 'all', [12])
    obj.set_input_data(
        "input_data",
        paddle.to_tensor(randtool("float", -1, 1, [4, 3, 10]).astype('bool')))
    obj.run()
