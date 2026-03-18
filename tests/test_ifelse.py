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


class BaseNet1(paddle.nn.Layer):
    def __init__(self):
        super(BaseNet1, self).__init__()

    def forward(self, inputs):
        if inputs == 1:
            return inputs * 1
        else:
            return inputs * 3


def test_ifelse_1_true():
    op = BaseNet1()
    op.eval()
    obj = APIOnnx(op, "ifelse", [11])
    obj.set_input_data("input_data", paddle.to_tensor(1))
    obj.run()


def test_ifelse_1_false():
    op = BaseNet1()
    op.eval()
    obj = APIOnnx(op, "ifelse", [11])
    obj.set_input_data("input_data", paddle.to_tensor(2))
    obj.run()


class BaseNet2(paddle.nn.Layer):
    def __init__(self):
        super(BaseNet2, self).__init__()

    def forward(self, cond, inputs):
        if cond == 1:
            return inputs * 1, inputs * 2
        else:
            return inputs * 3, inputs * 4


def test_ifelse_2_true():
    op = BaseNet2()
    op.eval()
    obj = APIOnnx(op, "ifelse", [11])
    obj.set_input_data("input_data", paddle.to_tensor(1), paddle.to_tensor(1))
    obj.run()


def test_ifelse_2_false():
    op = BaseNet2()
    op.eval()
    obj = APIOnnx(op, "ifelse", [11])
    obj.set_input_data("input_data", paddle.to_tensor(2), paddle.to_tensor(1))
    obj.run()


class BaseNet3(paddle.nn.Layer):
    def __init__(self):
        super(BaseNet3, self).__init__()

    def forward(self, inputs):
        if inputs == 1:
            return 1
        else:
            return 2


def test_ifelse_3_true():
    op = BaseNet3()
    op.eval()
    obj = APIOnnx(op, "ifelse", [11])
    obj.set_input_data("input_data", paddle.to_tensor(1))
    obj.run()


def test_ifelse_3_false():
    op = BaseNet3()
    op.eval()
    obj = APIOnnx(op, "ifelse", [11])
    obj.set_input_data("input_data", paddle.to_tensor(2))
    obj.run()


class BaseNet4(paddle.nn.Layer):
    def __init__(self):
        super(BaseNet4, self).__init__()

    def forward(self, inputs):
        if inputs == 1:
            return inputs + 1
        else:
            return 2


def test_ifelse_4_true():
    op = BaseNet4()
    op.eval()
    obj = APIOnnx(op, "ifelse", [11])
    obj.set_input_data("input_data", paddle.to_tensor(1))
    obj.run()


def test_ifelse_4_false():
    op = BaseNet4()
    op.eval()
    obj = APIOnnx(op, "ifelse", [11])
    obj.set_input_data("input_data", paddle.to_tensor(2))
    obj.run()


class BaseNet5(paddle.nn.Layer):
    def __init__(self):
        super(BaseNet5, self).__init__()

    def forward(self, inputs):
        if inputs == 1:
            return 1, 2
        else:
            return 2, 3


def test_ifelse_5_true():
    op = BaseNet5()
    op.eval()
    obj = APIOnnx(op, "ifelse", [11])
    obj.set_input_data("input_data", paddle.to_tensor(1))
    obj.run()


def test_ifelse_5_false():
    op = BaseNet5()
    op.eval()
    obj = APIOnnx(op, "ifelse", [11])
    obj.set_input_data("input_data", paddle.to_tensor(2))
    obj.run()


if __name__ == "__main__":
    test_ifelse_1_true()
    test_ifelse_1_false()
    test_ifelse_2_true()
    test_ifelse_2_false()
    test_ifelse_3_true()
    test_ifelse_3_false()
    test_ifelse_4_true()
    test_ifelse_4_false()
    test_ifelse_5_true()
    test_ifelse_5_false()
