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

    def __init__(self):
        super(Net, self).__init__()

    def forward(self, inputs, inputs_):
        """
        forward
        """
        x = paddle.greater_than(inputs, inputs_)
        print(x)
        return x


def test_greater_than_7():
    """
    api: paddle.greater_than
    op version: 7
    """
    op = Net()
    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, 'greater_than', [7])
    obj.set_input_data(
        "input_data",
        paddle.to_tensor(randtool("float", -1, 1, [3, 10]).astype('float32')),
        paddle.to_tensor(randtool("float", 0, 1, [3, 10]).astype('float32')))
    obj.run()




def test_greater_than_9():
    """
    api: paddle.greater_than
    op version: 9
    """
    op = Net()
    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, 'greater_than', [9])
    obj.set_input_data(
        "input_data",
        paddle.to_tensor(randtool("float", -1, 1, [3, 10]).astype('float32')),
        paddle.to_tensor(randtool("float", 0, 1, [3, 10]).astype('float32')))
    obj.run()

def test_greater_than_9_bool():
    """
    api: paddle.greater_than
    op version: 9
    """
    op = Net()
    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, 'greater_than', [9])
    obj.set_input_data(
        "input_data",
        paddle.to_tensor([True, False, True, False, True], dtype=paddle.bool),
        paddle.to_tensor([False], dtype=paddle.bool))
    obj.run()
    # x[0] = 1

def test_greater_than_9_bool_matrix():
    """
    api: paddle.greater_than
    op version: 9
    """
    op = Net()
    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, 'greater_than', [9])
    obj.set_input_data(
        "input_data",
        paddle.to_tensor([[True, False, True, False, True],[False, False, True, True, True]], dtype=paddle.bool),
        paddle.to_tensor([[False],[True]], dtype=paddle.bool))
    obj.run()
    # x[0] = 1