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
            out = inputs * 1
            # out2 = inputs *1
        else:
            out = inputs * 3
            # out2 = inputs * 3
        # return (out,out2)
        return out


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
            # return inputs * 1, inputs * 2
            out = inputs * 1
            out1 = inputs * 2
        else:
            # return inputs * 3, inputs * 4
            out = inputs * 1
            out1 = inputs * 2
        return (out, out1)


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


if __name__ == "__main__":
    test_ifelse_1_true()
    test_ifelse_1_false()
    test_ifelse_2_true()
    test_ifelse_2_false()
