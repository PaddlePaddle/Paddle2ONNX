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
    def forward(self, shape):
       # assert(type(value) == bool)
        x = paddle.full(shape=shape, fill_value=False, dtype=paddle.bool)
        print(x)
        return x


def test_flatten_9_bool():
    """
    api: paddle.fill_constant
    op version: 9
    """
    op = Net()
    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, 'fill_constant', [9])
    obj.set_input_data(
        "input_data",
        paddle.to_tensor([2,3], dtype=paddle.int64),
    )
    obj.run()