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
from onnxbase import _test_with_pir


class Net(paddle.nn.Layer):
    """
    simple Net
    """

    def __init__(self):
        super(Net, self).__init__()

    def forward(self, inputs, inputs_):
        """
        forward
        """
        x = paddle.matmul(inputs, inputs_, transpose_x=False, transpose_y=True)
        return x


@_test_with_pir
def test_matmul_9():
    """
    api: paddle.matmul
    op version: 9
    """
    op = Net()
    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, 'matmul', [9])
    obj.set_input_data(
        "input_data",
        paddle.to_tensor(randtool("float", -1, 1, [3, 10]).astype('float32')),
        paddle.to_tensor(randtool("float", 0, 1, [3, 10]).astype('float32')))
    obj.run()


@_test_with_pir
def test_matmul_10():
    """
    api: paddle.matmul
    op version: 9
    """
    op = Net()
    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, 'matmul', [10])
    obj.set_input_data(
        "input_data",
        paddle.to_tensor(randtool("float", -1, 1, [3, 10]).astype('float32')),
        paddle.to_tensor(randtool("float", 0, 1, [3, 10]).astype('float32')))
    obj.run()


@_test_with_pir
def test_matmul_11():
    """
    api: paddle.matmul
    op version: 11
    """
    op = Net()
    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, 'matmul', [11])
    obj.set_input_data(
        "input_data",
        paddle.to_tensor(randtool("float", -1, 1, [3, 10]).astype('float32')),
        paddle.to_tensor(randtool("float", 0, 1, [3, 10]).astype('float32')))
    obj.run()


@_test_with_pir
def test_matmul_12():
    """
    api: paddle.matmul
    op version: 12
    """
    op = Net()
    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, 'matmul', [12])
    obj.set_input_data(
        "input_data",
        paddle.to_tensor(randtool("float", -1, 1, [3, 10]).astype('float32')),
        paddle.to_tensor(randtool("float", 0, 1, [3, 10]).astype('float32')))
    obj.run()
