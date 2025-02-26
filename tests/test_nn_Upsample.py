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
from onnxbase import APIOnnx, randtool, _test_with_pir


class Net(paddle.nn.Layer):
    """
    simple Net
    """

    def __init__(
        self,
        size=None,
        scale_factor=None,
        mode="nearest",
        align_corners=False,
        align_mode=0,
        data_format="NCHW",
    ):
        super(Net, self).__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners
        self.align_mode = align_mode
        self.data_format = data_format

    def forward(self, inputs):
        """
        forward
        """
        x = paddle.nn.functional.upsample(
            x=inputs,
            size=self.size,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=self.align_corners,
            align_mode=self.align_mode,
            data_format=self.data_format,
        )
        return x


@_test_with_pir
def test_Upsample_size():
    """
    api: paddle.nn.functional.upsample
    op version: 11, 12
    """
    op = Net(size=[12, 12], align_mode=1)

    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, "nn_Upsample", [11, 12])
    obj.set_input_data(
        "input_data",
        paddle.to_tensor(randtool("float", -1, 1, [3, 1, 10, 10]).astype("float32")),
    )
    obj.run()


# has a bug
# def test_Upsample_size_tensor():
#     """
#     api: paddle.nn.functional.upsample
#     op version: 11, 12
#     """
#     op = Net(scale_factor=(paddle.to_tensor(2), paddle.to_tensor(2)),
#              align_mode=1)
#
#     op.eval()
#     # net, name, ver_list, delta=1e-6, rtol=1e-5
#     obj = APIOnnx(op, 'nn_Upsample', [11, 12])
#     obj.set_input_data(
#         "input_data",
#         paddle.to_tensor(
#             randtool("float", -1, 1, [3, 1, 10, 10]).astype('float32')))
#     obj.run()


@_test_with_pir
def test_Upsample_scale_factor():
    """
    api: paddle.nn.functional.upsample
    op version: 11, 12
    """
    op = Net(scale_factor=[2, 3])

    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, "nn_Upsample", [11, 12])
    obj.set_input_data(
        "input_data",
        paddle.to_tensor(randtool("float", -1, 1, [3, 1, 10, 10]).astype("float32")),
    )
    obj.run()


@_test_with_pir
def test_Upsample_size_linear_tensor():
    """
    api: paddle.nn.functional.upsample
    op version: 11, 12
    """
    op = Net(size=[12], mode="linear", data_format="NCW", align_mode=1)

    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, "nn_Upsample", [11])
    obj.set_input_data(
        "input_data",
        paddle.to_tensor(randtool("float", -1, 1, [1, 1, 10]).astype("float32")),
    )
    obj.run()


@_test_with_pir
def test_Upsample_size_linear():
    """
    api: paddle.nn.functional.upsample
    op version: 11, 12
    """
    op = Net(
        size=[12], mode="linear", align_corners=False, align_mode=1, data_format="NCW"
    )

    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, "nn_Upsample", [11, 12])
    obj.set_input_data(
        "input_data",
        paddle.to_tensor(randtool("float", -1, 1, [1, 1, 10]).astype("float32")),
    )
    obj.run()


@_test_with_pir
def test_Upsample_scale_factor_linear():
    """
    api: paddle.nn.functional.upsample
    op version: 11, 12
    """
    op = Net(
        scale_factor=[1.5],
        mode="linear",
        align_corners=False,
        align_mode=1,
        data_format="NCW",
    )

    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, "nn_Upsample", [11, 12])
    obj.set_input_data(
        "input_data",
        paddle.to_tensor(randtool("float", -1, 1, [1, 1, 10]).astype("float32")),
    )
    obj.run()


@_test_with_pir
def test_Upsample_size_bilinear():
    """
    api: paddle.nn.functional.upsample
    op version: 11, 12
    """
    op = Net(
        size=[12, 15],
        mode="bilinear",
        align_corners=False,
        align_mode=1,
        data_format="NCHW",
    )

    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, "nn_Upsample", [11, 12])
    obj.set_input_data(
        "input_data",
        paddle.to_tensor(randtool("float", -1, 1, [3, 1, 10, 10]).astype("float32")),
    )
    obj.run()


@_test_with_pir
def test_Upsample_scale_factor_bilinear():
    """
    api: paddle.nn.functional.upsample
    op version: 11, 12
    """
    op = Net(
        scale_factor=[2, 3],
        mode="bilinear",
        align_corners=False,
        align_mode=1,
        data_format="NCHW",
    )

    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, "nn_Upsample", [11, 12])
    obj.set_input_data(
        "input_data",
        paddle.to_tensor(randtool("float", -1, 1, [3, 1, 10, 10]).astype("float32")),
    )
    obj.run()


@_test_with_pir
def test_Upsample_size_nearest():
    """
    api: paddle.nn.functional.upsample
    op version: 11, 12
    """
    op = Net(
        size=[12, 15],
        mode="nearest",
        align_corners=False,
        align_mode=1,
        data_format="NCHW",
    )

    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, "nn_Upsample", [11, 12])
    obj.set_input_data(
        "input_data",
        paddle.to_tensor(randtool("float", -1, 1, [3, 1, 10, 10]).astype("float32")),
    )
    obj.run()


@_test_with_pir
def test_Upsample_scale_factor_nearest():
    """
    api: paddle.nn.functional.upsample
    op version: 11, 12
    """
    op = Net(
        scale_factor=[2, 3],
        mode="nearest",
        align_corners=False,
        align_mode=1,
        data_format="NCHW",
    )

    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, "nn_Upsample", [11, 12])
    obj.set_input_data(
        "input_data",
        paddle.to_tensor(randtool("float", -1, 1, [3, 1, 10, 10]).astype("float32")),
    )
    obj.run()


@_test_with_pir
def test_Upsample_size_bicubic():
    """
    api: paddle.nn.functional.upsample
    op version: 11, 12
    """
    op = Net(
        size=[12, 15],
        mode="bicubic",
        align_corners=False,
        align_mode=1,
        data_format="NCHW",
    )

    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, "nn_Upsample", [11, 12])
    obj.set_input_data(
        "input_data",
        paddle.to_tensor(randtool("float", -1, 1, [3, 1, 10, 10]).astype("float32")),
    )
    obj.run()


@_test_with_pir
def test_Upsample_scale_factor_bicubic():
    """
    api: paddle.nn.functional.upsample
    op version: 11, 12
    """
    op = Net(
        scale_factor=[2, 3],
        mode="bicubic",
        align_corners=False,
        align_mode=1,
        data_format="NCHW",
    )

    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, "nn_Upsample", [11, 12])
    obj.set_input_data(
        "input_data",
        paddle.to_tensor(randtool("float", -1, 1, [3, 1, 10, 10]).astype("float32")),
    )
    obj.run()


@_test_with_pir
def test_Upsample_size_trilinear():
    """
    api: paddle.nn.functional.upsample
    op version: 11, 12
    """
    op = Net(
        size=[12, 15, 20],
        mode="trilinear",
        align_corners=False,
        align_mode=1,
        data_format="NCDHW",
    )

    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, "nn_Upsample", [11, 12])
    obj.set_input_data(
        "input_data",
        paddle.to_tensor(
            randtool("float", -1, 1, [3, 1, 10, 10, 10]).astype("float32")
        ),
    )
    obj.run()


@_test_with_pir
def test_Upsample_scale_factor_trilinear():
    """
    api: paddle.nn.functional.upsample
    op version: 11, 12
    """
    op = Net(
        scale_factor=[2, 3, 4],
        mode="trilinear",
        align_corners=False,
        align_mode=1,
        data_format="NCDHW",
    )

    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, "nn_Upsample", [11, 12])
    obj.set_input_data(
        "input_data",
        paddle.to_tensor(
            randtool("float", -1, 1, [3, 1, 10, 10, 10]).astype("float32")
        ),
    )
    obj.run()


if __name__ == "__main__":
    test_Upsample_size()
    test_Upsample_scale_factor()
    test_Upsample_size_linear_tensor()
    test_Upsample_size_linear()
    test_Upsample_scale_factor_linear()
    test_Upsample_size_bilinear()
    test_Upsample_scale_factor_bilinear()
    test_Upsample_size_nearest()
    test_Upsample_scale_factor_nearest()
    test_Upsample_size_bicubic()
    test_Upsample_scale_factor_bicubic()
    test_Upsample_size_trilinear()
    test_Upsample_scale_factor_trilinear()
