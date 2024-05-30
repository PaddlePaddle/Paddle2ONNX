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
        x = paddle.bitwise_and(inputs, inputs_)
        return x


def test_bitwise_and_18_1():
    """
    api: paddle.bitwise_and
    op version: 18
    """
    op = Net()
    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, 'BitwiseAnd', [18])
    obj.set_input_data(
        "input_data",
        paddle.to_tensor([-5, -1, 1]),
        paddle.to_tensor([4,  2, -3]))
    obj.run()

def test_bitwise_and_18_2():
    """
    api: paddle.bitwise_and
    op version: 18
    """
    op = Net()
    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, 'BitwiseAnd', [18])
    obj.set_input_data(
        "input_data",
        paddle.to_tensor([True, True, True]),
        paddle.to_tensor([False,  False, True]))
    obj.run()
def test_bitwise_and_18_3():
    """
    api: paddle.bitwise_and
    op version: 18
    """
    op = Net()
    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, 'BitwiseAnd', [18])
    a = paddle.to_tensor([True,  False, True])
    b = paddle.to_tensor([-5, -1, 1])
    print(type(a[0].item()))

    obj.set_input_data(
        "input_data",
        a,
        b
    )
    obj.run()

if __name__ == '__main__':
    test_bitwise_and_18_2()