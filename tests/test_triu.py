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
import numpy as np

class Net(paddle.nn.Layer):
    """
    simple Net
    """

    def __init__(self):
        super(Net, self).__init__()

    def forward(self, inputs, diagonal):
        """
        forward
        """
        x = paddle.triu(inputs, diagonal=diagonal)
        return x


def test_triu_14_1():
    """
    api: paddle.triu
    op version: 14
    """
    op = Net()
    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, 'tril_triu', [14])
    input_data = paddle.to_tensor(randtool("float", -1, 1, [4,5]).astype('float32'))
    diagonal = paddle.to_tensor(3).astype('int64')
    print(input_data)
    obj.set_input_data(
        "input_data",
        input_data,
        diagonal,
        )
    obj.run()

# def test_triu_14_2():
#     """
#     api: paddle.triu
#     op version: 14
#     """
#     op = Net()
#     op.eval()
#     # net, name, ver_list, delta=1e-6, rtol=1e-5
#     obj = APIOnnx(op, 'tril_triu', [14])
#     input_data = paddle.to_tensor(randtool("float", -1, 1, [4,5]).astype('float32'))
#     diagonal = 1;
#     print(input_data)
#     obj.set_input_data(
#         "input_data",
#         input_data,
#         diagonal
#         )
#     obj.run()

# def test_triu_14_3():
#     """
#     api: paddle.triu
#     op version: 14
#     """
#     op = Net()
#     op.eval()
#     # net, name, ver_list, delta=1e-6, rtol=1e-5
#     obj = APIOnnx(op, 'tril_triu', [14])
#     input_data = paddle.to_tensor(randtool("float", -1, 1, [4,5]).astype('float32'))
#     diagonal = 2;
#     print(input_data)
#     obj.set_input_data(
#         "input_data",
#         input_data,
#         diagonal
#         )
#     obj.run()
# def test_triu_14_4():
#     """
#     api: paddle.triu
#     op version: 14
#     """
#     op = Net()
#     op.eval()
#     # net, name, ver_list, delta=1e-6, rtol=1e-5
#     obj = APIOnnx(op, 'tril_triu', [14])
#     input_data = paddle.to_tensor(randtool("float", -1, 1, [4,5]).astype('float32'))
#     diagonal = 3;
#     print(input_data)
#     obj.set_input_data(
#         "input_data",
#         input_data,
#         diagonal
#         )
#     obj.run()
if __name__ == "__main__":
    test_triu()