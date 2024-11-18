# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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


class BaseNet1(paddle.nn.Layer):
    def __init__(self):
        super(BaseNet1, self).__init__()

    def forward(self, inputs):
        i = 0
        while i <= 3:
            i += 1
            inputs += 1
        return inputs


@_test_only_pir
def test_while_1():
    op = BaseNet1()
    op.eval()
    obj = APIOnnx(op, "while", [13])
    obj.set_input_data("input_data", paddle.to_tensor(0))
    obj.run()


class BaseNet2(paddle.nn.Layer):
    def __init__(self):
        super(BaseNet2, self).__init__()

    def forward(self, i, inputs):
        while i <= 3:
            i += 1
            inputs += 1
        return inputs


@_test_only_pir
def test_while_2():
    op = BaseNet2()
    op.eval()
    obj = APIOnnx(op, "while", [13])
    obj.set_input_data("input_data", paddle.to_tensor(0), paddle.to_tensor(0))
    obj.run()


class BaseNet3(paddle.nn.Layer):
    def __init__(self):
        super(BaseNet3, self).__init__()

    def forward(self, i, j, k):
        while i <= 3:
            j += 1
            k += 1
            i += 1
        return j + k


@_test_only_pir
def test_while_3():
    op = BaseNet3()
    op.eval()
    obj = APIOnnx(op, "while", [13])
    obj.set_input_data(
        "input_data", paddle.to_tensor(0), paddle.to_tensor(0), paddle.to_tensor(0)
    )
    obj.run()


class BaseNet4(paddle.nn.Layer):
    def __init__(self):
        super(BaseNet4, self).__init__()

    def forward(self, i, j, k):
        while i <= 3:
            if i < 1:
                j += 1
            else:
                j += 2
            i += 1
        return j + k


@_test_only_pir
def test_while_4():
    op = BaseNet4()
    op.eval()
    obj = APIOnnx(op, "while", [13])
    obj.set_input_data(
        "input_data", paddle.to_tensor(0), paddle.to_tensor(0), paddle.to_tensor(0)
    )
    obj.run()


if __name__ == "__main__":
    test_while_1()
    test_while_2()
    test_while_3()
    test_while_4()
