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
from onnxbase import _test_only_pir


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
        x = paddle.gather(
            inputs,
            index=paddle.to_tensor([1, 2], dtype="int64"),
            axis=paddle.to_tensor([0]),
        )
        return x


@_test_only_pir
def test_gather_7():
    """
    api: paddle.gather
    op version: 7
    """
    op = Net()
    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, "gather", [7])
    obj.set_input_data(
        "input_data",
        paddle.to_tensor(randtool("float", -1, 1, [3, 10]).astype("float32")),
    )
    obj.run()


@_test_only_pir
def test_gather_11():
    """
    api: paddle.gather
    op version: 11
    """
    op = Net()
    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, "gather", [13])
    obj.set_input_data(
        "input_data",
        paddle.to_tensor(randtool("float", -1, 1, [3, 10]).astype("float32")),
    )
    obj.run()


@_test_only_pir
def test_gather_13():
    """
    api: paddle.gather
    op version: 13
    """
    op = Net()
    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, "gather", [13])
    obj.set_input_data(
        "input_data",
        paddle.to_tensor(randtool("float", -1, 1, [3, 10]).astype("float32")),
    )
    obj.run()


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
        x = paddle.gather(
            inputs, index=paddle.to_tensor([[1], [2]], dtype="int64"), axis=1
        )
        return x


# Attention : GatherND don't have opset < 11 version, so we don't test it.


@_test_only_pir
def test_gather_11_2():
    """
    api: paddle.gather
    op version: 11
    """
    op = Net2()
    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, "gather_2", [11])
    # data: [batch_size, mat_h, mat_w], index: [idx_size, 1]
    data = paddle.to_tensor(randtool("float", -1, 1, [1, 6, 8]).astype("float32"))
    print(data.shape)
    obj.set_input_data("input_data", data)
    obj.run()


@_test_only_pir
def test_gather_13_2():
    """
    api: paddle.gather
    op version: 13
    """
    op = Net2()
    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, "gather_2", [13])
    # data: [batch_size, mat_h, mat_w], index: [idx_size, 1]
    data = paddle.to_tensor(randtool("float", -1, 1, [1, 6, 8]).astype("float32"))
    print(data.shape)
    obj.set_input_data("input_data", data)
    obj.run()


class Net3(paddle.nn.Layer):
    """
    simple Net
    """

    def __init__(self):
        super(Net3, self).__init__()

    def forward(self, inputs):
        """
        forward
        """
        x = paddle.gather(inputs, index=paddle.to_tensor([0, 1], dtype="int64"), axis=1)
        return x


@_test_only_pir
def test_gather_7_3():
    """
    api: paddle.gather
    op version: 7
    """
    op = Net3()
    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, "gather_3", [7])
    # data: [batch_size, mat_h, mat_w], index: [idx_size, 1]
    data = paddle.to_tensor(randtool("float", -1, 1, [1, 6, 8]).astype("float32"))
    print(data.shape)
    obj.set_input_data("input_data", data)
    obj.run()
    assert len(obj.res_fict["7"][0].shape) == len(
        data.shape
    ), "The result of ONNX inference is not equal to Paddle inference!\n"


@_test_only_pir
def test_gather_11_3():
    """
    api: paddle.gather
    op version: 11
    """
    op = Net3()
    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, "gather_3", [11])
    # data: [batch_size, mat_h, mat_w], index: [idx_size, 1]
    data = paddle.to_tensor(randtool("float", -1, 1, [1, 6, 8]).astype("float32"))
    print(data.shape)
    obj.set_input_data("input_data", data)
    obj.run()
    assert len(obj.res_fict["11"][0].shape) == len(
        data.shape
    ), "The result of ONNX inference is not equal to Paddle inference!\n"


@_test_only_pir
def test_gather_13_3():
    """
    api: paddle.gather
    op version: 13
    """
    op = Net3()
    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, "gather_3", [13])
    # data: [batch_size, mat_h, mat_w], index: [idx_size, 1]
    data = paddle.to_tensor(randtool("float", -1, 1, [1, 6, 8]).astype("float32"))
    print(data.shape)
    obj.set_input_data("input_data", data)

    obj.run()
    assert len(obj.res_fict["13"][0].shape) == len(
        data.shape
    ), "The result of ONNX inference is not equal to Paddle inference!\n"


class Net4(paddle.nn.Layer):
    """
    simple Net
    """

    def __init__(self):
        super(Net4, self).__init__()

    def forward(self, inputs):
        """
        forward
        """
        x = paddle.gather(
            inputs, index=paddle.to_tensor([[0], [1]], dtype="int64"), axis=2
        )
        return x


@_test_only_pir
def test_gather_11_4():
    """
    api: paddle.gather
    op version: 11
    """
    op = Net4()
    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, "gather_4", [11])
    # data: [batch_size, mat_h, mat_w], index: [idx_size, 1]
    data = paddle.to_tensor(randtool("float", -1, 1, [1, 6, 8]).astype("float32"))
    print(data.shape)
    obj.set_input_data("input_data", data)
    obj.run()


@_test_only_pir
def test_gather_13_4():
    """
    api: paddle.gather
    op version: 13
    """
    op = Net4()
    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, "gather_4", [13])
    # data: [batch_size, mat_h, mat_w], index: [idx_size, 1]
    data = paddle.to_tensor(randtool("float", -1, 1, [1, 6, 8]).astype("float32"))
    print(len(data.shape))
    obj.set_input_data("input_data", data)
    obj.run()
    assert len(obj.res_fict["13"][0].shape) == len(
        data.shape
    ), "The result of ONNX inference is not equal to Paddle inference!\n"
