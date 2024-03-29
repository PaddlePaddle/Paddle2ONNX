# Copyright (c) 2022  PaddlePaddle Authors. All Rights Reserved.
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
from caffebase import APICaffe
from caffebase import randtool


class Net(paddle.nn.Layer):
    """
    simple Net
    """

    def __init__(self,
                 kernel_size=2,
                 stride=None,
                 padding=0,
                 return_mask=False,
                 ceil_mode=False,
                 data_format="NCHW",
                 name=None):
        super(Net, self).__init__()
        self._max_pool = paddle.nn.MaxPool2D(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            return_mask=return_mask,
            ceil_mode=ceil_mode,
            data_format=data_format,
            name=name)

    def forward(self, inputs):
        """
        forward
        """
        x = self._max_pool(inputs)
        return x


common_shape = [3, 3, 20, 20]


def test_MaxPool2D_base():
    """
    api: paddle.MaxPool2D
    """
    op = Net()
    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APICaffe(op, 'nn_MaxPool2D')
    obj.set_input_data(
        "input_data",
        paddle.to_tensor(
            randtool("float", -1, 1, common_shape).astype('float32')))
    obj.run()


def test_MaxPool2D_base_VALID():
    """
    api: paddle.MaxPool2D
    """
    op = Net(kernel_size=5, stride=3, padding='VALID')
    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APICaffe(op, 'nn_MaxPool2D')
    obj.set_input_data(
        "input_data",
        paddle.to_tensor(
            randtool("float", -1, 1, common_shape).astype('float32')))
    obj.run()


def test_MaxPool2D_base_SAME():
    """
    api: paddle.MaxPool2D
    """
    op = Net(kernel_size=5, padding='SAME')
    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APICaffe(op, 'nn_MaxPool2D')
    obj.set_input_data(
        "input_data",
        paddle.to_tensor(
            randtool("float", -1, 1, common_shape).astype('float32')))
    obj.run()


def test_MaxPool2D_base_Padding_1():
    """
    api: paddle.MaxPool2D
    """
    op = Net(kernel_size=5, padding=[1, 2, 3, 4])
    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APICaffe(op, 'nn_MaxPool2D')
    obj.set_input_data(
        "input_data",
        paddle.to_tensor(
            randtool("float", -1, 1, common_shape).astype('float32')))
    obj.run()


def test_MaxPool2D_base_Padding_2():
    """
    api: paddle.MaxPool2D
    """
    op = Net(kernel_size=3, padding=[1, 2])
    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APICaffe(op, 'nn_MaxPool2D')
    obj.set_input_data(
        "input_data",
        paddle.to_tensor(
            randtool("float", -1, 1, common_shape).astype('float32')))
    obj.run()


def test_MaxPool2D_base_Padding_3():
    """
    api: paddle.MaxPool2D
    """
    op = Net(kernel_size=5, padding=2)
    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APICaffe(op, 'nn_MaxPool2D')
    obj.set_input_data(
        "input_data",
        paddle.to_tensor(
            randtool("float", -1, 1, common_shape).astype('float32')))
    obj.run()


if __name__ == '__main__':
    # test_MaxPool2D_base()
    # test_MaxPool2D_base_SAME()
    # test_MaxPool2D_base_VALID()
    # test_MaxPool2D_base_Padding_3()
    test_MaxPool2D_base_Padding_2()
    # test_MaxPool2D_base_Padding_1()  TODO
