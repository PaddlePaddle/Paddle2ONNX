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
from onnxbase import _test_only_pir


class Net1(paddle.nn.Layer):
    """
    simple Net
    """

    def __init__(self):
        super(Net1, self).__init__()

    def forward(self, shape):
        """
        forward
        """
        x = paddle.full(shape, fill_value=3)
        return x


@_test_only_pir
def test_full_with_tensor_1():
    """
    op version: 8
    """
    op = Net1()
    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, "full_with_tensor", [8])
    obj.set_input_data("input_data", paddle.to_tensor([2, 3]))
    obj.run()


class Net2(paddle.nn.Layer):
    """
    simple Net
    """

    def __init__(self):
        super(Net2, self).__init__()

    def forward(self, shape, fill_value):
        """
        forward
        """
        x = paddle.full(shape, fill_value)
        return x


@_test_only_pir
def test_full_with_tensor_2():
    """
    op version: 8
    """
    op = Net2()
    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, "full_with_tensor", [8])
    obj.set_input_data("input_data", paddle.to_tensor([2, 3]), paddle.to_tensor(3))
    obj.run()


@_test_only_pir
def test_full_with_tensor_3():
    """
    op version: 8
    """
    op = Net2()
    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, "full_with_tensor", [8])
    obj.set_input_data(
        "input_data", paddle.ones(shape=[3, 2]).astype("int32"), paddle.to_tensor(3)
    )
    obj.run()
