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
from onnxbase import APIOnnx, randtool
from onnxbase import _test_with_pir


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
        x = paddle.isinf(inputs)
        return x.astype("float32")


@_test_with_pir
def test_isnan_int32():
    """
    api: paddle.isnan
    op version: 9
    """
    op = Net()
    op.eval()

    obj = APIOnnx(op, "isinf", [10])

    obj.set_input_data(
        "input_data",
        paddle.to_tensor(
            paddle.to_tensor(randtool("int", -10, 20, [3, 3, 3]).astype("int32")),
        ),
    )

    obj.run()


@_test_with_pir
def test_isinf_10():
    """
    api: paddle.isinf
    op version: 10
    """
    op = Net()
    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, "isinf", [10])
    obj.set_input_data(
        "input_data",
        paddle.to_tensor(
            ([float("-inf"), -2, 3.6, float("inf"), 0, float("-nan"), float("nan")])
        ),
    )
    obj.run()


@_test_with_pir
def test_isinf_11():
    """
    api: paddle.isinf
    op version: 11
    """
    op = Net()
    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, "isinf", [11])
    obj.set_input_data(
        "input_data",
        paddle.to_tensor(
            ([float("-inf"), -2, 3.6, float("inf"), 0, float("-nan"), float("nan")])
        ),
    )
    obj.run()


@_test_with_pir
def test_isinf_12():
    """
    api: paddle.isinf
    op version: 12
    """
    op = Net()
    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, "isinf", [12])
    obj.set_input_data(
        "input_data",
        paddle.to_tensor(
            ([float("-inf"), -2, 3.6, float("inf"), 0, float("-nan"), float("nan")])
        ),
    )
    obj.run()
