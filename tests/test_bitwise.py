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
from onnxbase import _test_with_pir


class BitwiseAndNet(paddle.nn.Layer):
    def __init__(self):
        super(BitwiseAndNet, self).__init__()

    def forward(self, x, y):
        x = paddle.bitwise_and(x, y)
        return x


@_test_with_pir
def test_bitwise_and_int_type_18():
    """
    api: paddle.bitwise_and
    op version: 18
    """
    op = BitwiseAndNet()
    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, "BitwiseAnd", [18])
    obj.set_input_data(
        "input_data", paddle.to_tensor([-5, -1, 1]), paddle.to_tensor([4, 2, -3])
    )
    obj.run()


@_test_with_pir
def test_bitwise_and_bool_type():
    """
    api: paddle.bitwise_and
    op version: 7
    """
    op = BitwiseAndNet()
    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, "BitwiseAnd", [7])
    obj.set_input_data(
        "input_data",
        paddle.to_tensor([True, True, True]),
        paddle.to_tensor([False, False, True]),
    )
    obj.run()


@_test_with_pir
def test_bitwise_and_bool_type_18():
    """
    api: paddle.bitwise_and
    op version: 18
    """
    op = BitwiseAndNet()
    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, "BitwiseAnd", [18])
    obj.set_input_data(
        "input_data",
        paddle.to_tensor([True, True, True]),
        paddle.to_tensor([False, False, True]),
    )
    obj.run()


class BitwiseNotNet(paddle.nn.Layer):
    def __init__(self):
        super(BitwiseNotNet, self).__init__()

    def forward(self, x):
        x = paddle.bitwise_not(x)
        return x


@_test_with_pir
def test_bitwise_not_int_type_18():
    """
    api: paddle.bitwise_not
    op version: 18
    """
    op = BitwiseNotNet()
    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, "BitwiseNot", [18])
    obj.set_input_data("input_data", paddle.to_tensor([-5, -1, 1]))
    obj.run()


@_test_with_pir
def test_bitwise_not_bool_type():
    """
    api: paddle.bitwise_not
    op version: 7
    """
    op = BitwiseNotNet()
    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, "BitwiseNot", [7])
    obj.set_input_data("input_data", paddle.to_tensor([True, True, True]))
    obj.run()


@_test_with_pir
def test_bitwise_not_bool_type_18():
    """
    api: paddle.bitwise_not
    op version: 18
    """
    op = BitwiseNotNet()
    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, "BitwiseNot", [18])
    obj.set_input_data("input_data", paddle.to_tensor([True, True, True]))
    obj.run()


class BitwiseOrNet(paddle.nn.Layer):
    def __init__(self):
        super(BitwiseOrNet, self).__init__()

    def forward(self, x, y):
        x = paddle.bitwise_or(x, y)
        return x


@_test_with_pir
def test_bitwise_or_int_type_18():
    """
    api: paddle.bitwise_or
    op version: 18
    """
    op = BitwiseOrNet()
    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, "BitwiseOr", [18])
    obj.set_input_data(
        "input_data", paddle.to_tensor([-5, -1, 1]), paddle.to_tensor([4, 2, -3])
    )
    obj.run()


@_test_with_pir
def test_bitwise_or_bool_type():
    """
    api: paddle.bitwise_or
    op version: 7
    """
    op = BitwiseOrNet()
    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, "BitwiseOr", [7])
    obj.set_input_data(
        "input_data",
        paddle.to_tensor([True, True, True]),
        paddle.to_tensor([False, False, True]),
    )
    obj.run()


@_test_with_pir
def test_bitwise_or_bool_type_18():
    """
    api: paddle.bitwise_or
    op version: 18
    """
    op = BitwiseOrNet()
    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, "BitwiseOr", [18])
    obj.set_input_data(
        "input_data",
        paddle.to_tensor([True, True, True]),
        paddle.to_tensor([False, False, True]),
    )
    obj.run()


class BitwiseXorNet(paddle.nn.Layer):
    def __init__(self):
        super(BitwiseXorNet, self).__init__()

    def forward(self, x, y):
        x = paddle.bitwise_xor(x, y)
        return x


@_test_with_pir
def test_bitwise_xor_int_type_18():
    """
    api: paddle.bitwise_xor
    op version: 18
    """
    op = BitwiseXorNet()
    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, "BitwiseXor", [18])
    obj.set_input_data(
        "input_data", paddle.to_tensor([-5, -1, 1]), paddle.to_tensor([4, 2, -3])
    )
    obj.run()


@_test_with_pir
def test_bitwise_xor_bool_type():
    """
    api: paddle.bitwise_xor
    op version: 7
    """
    op = BitwiseXorNet()
    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, "BitwiseXor", [7])
    obj.set_input_data(
        "input_data",
        paddle.to_tensor([True, True, True]),
        paddle.to_tensor([False, False, True]),
    )
    obj.run()


@_test_with_pir
def test_bitwise_xor_bool_type_18():
    """
    api: paddle.bitwise_xor
    op version: 18
    """
    op = BitwiseXorNet()
    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, "BitwiseXor", [18])
    obj.set_input_data(
        "input_data",
        paddle.to_tensor([True, True, True]),
        paddle.to_tensor([False, False, True]),
    )
    obj.run()


if __name__ == "__main__":
    test_bitwise_not_int_type_18()
