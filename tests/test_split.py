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

    def forward(self, inputs):
        """
        forward
        """
        x = paddle.split(inputs, num_or_sections=5, axis=1)
        return x

class Net2(paddle.nn.Layer):
    """
    simple Net
    """

    def __init__(self):
        super(Net2, self).__init__()

    def forward(self, inputs):
        """
        forward
        """
        x = paddle.split(inputs, num_or_sections=[2,3,5], axis=-1)
        return x
    
def test_split_v7_1():
    """
    api: paddle.split
    op version: 7
    """
    op = Net()
    op.eval()
    obj = APIOnnx(op, 'split', [7])
    obj.set_input_data("input_data",
                       paddle.to_tensor(
                           randtool("float", -1, 1, [3, 10]).astype('float32')))
    obj.run()




def test_split_v7_2():
    """
    api: paddle.split
    op version: 7
    """
    op = Net2()
    op.eval()
    obj = APIOnnx(op, 'split2', [7])
    obj.set_input_data("input_data",
                       paddle.to_tensor(
                           randtool("float", -1, 1, [3, 10]).astype('float32')))
    obj.run()

def test_split_v13_1():
    """
    api: paddle.split
    op version: 13
    """
    op = Net()
    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, 'split', [13])
    obj.set_input_data("input_data",
                       paddle.to_tensor(
                           randtool("float", -1, 1, [3, 10]).astype('float32')))
    obj.run()


def test_split_v13_2():
    """
    api: paddle.split
    op version: 13
    """
    op = Net2()
    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, 'split', [13])
    obj.set_input_data("input_data",
                       paddle.to_tensor(
                           randtool("float", -1, 1, [3, 10]).astype('float32')))
    obj.run()

def test_split_v18_1():
    """
    api: paddle.split
    op version: 18
    """
    op = Net()
    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, 'split', [18])
    obj.set_input_data("input_data",
                       paddle.to_tensor(
                           randtool("float", -1, 1, [3, 10]).astype('float32')))
    obj.run()


def test_split_v18_2():
    """
    api: paddle.split
    op version: 18
    """
    op = Net2()
    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, 'split', [18])
    obj.set_input_data("input_data",
                       paddle.to_tensor(
                           randtool("float", -1, 1, [3, 10]).astype('float32')))
    obj.run()