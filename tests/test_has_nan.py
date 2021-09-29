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
    simplr Net
    """

    def __init__(self):
        super(Net, self).__init__()

    def forward(self, inputs):
        """
        forward
        """
        x = paddle.fluid.layers.has_nan(inputs)
        return x.astype('float32')


def test_has_nan_base():
    """
    api: paddle.fluid.layers.has_nan
    op version: 9, 10, 12
    """
    op = Net()
    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, 'has_nan', [9, 10, 12])
    obj.set_input_data(
        "input_data",
        paddle.to_tensor(([
            float('-inf'), -2, 3.6,
            float('inf'), 0,
            float('-nan'),
            float('nan')
        ])))
    obj.run()
