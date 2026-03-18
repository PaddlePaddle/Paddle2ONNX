# Copyright (c) 2024  PaddlePaddle Authors. All Rights Reserved.
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


@_test_with_pir
def test_einsum_sum():
    """
    api: paddle.einsum
    op version: 13
    """

    class Net(paddle.nn.Layer):
        """
        simple Net
        """

        def __init__(self):
            super(Net, self).__init__()

        def forward(self, input):
            """
            forward
            """
            x = paddle.einsum("i->", input)
            return x

    op = Net()
    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, "einsum_sum", [13])
    obj.set_input_data("input_data", paddle.rand([4]))
    obj.run()


@_test_with_pir
def test_einsum_dot():
    """
    api: paddle.einsum
    op version: 13
    """

    class Net(paddle.nn.Layer):
        """
        simple Net
        """

        def __init__(self):
            super(Net, self).__init__()

        def forward(self, x, y):
            """
            forward
            """
            x = paddle.einsum("i,i->", x, y)
            return x

    op = Net()
    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, "einsum_dot", [13])
    input_x = paddle.rand([4])
    obj.set_input_data("input_data", input_x, input_x)
    obj.run()


@_test_with_pir
def test_einsum_outer():
    """
    api: paddle.einsum
    op version: 13
    """

    class Net(paddle.nn.Layer):
        """
        simple Net
        """

        def __init__(self):
            super(Net, self).__init__()

        def forward(self, x, y):
            """
            forward
            """
            x = paddle.einsum("i,j->ij", x, y)
            return x

    op = Net()
    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, "einsum_dot", [13])
    input_x = paddle.rand([4])
    input_y = paddle.rand([5])
    obj.set_input_data("input_data", input_x, input_y)
    obj.run()


@_test_with_pir
def test_einsum_transpose():
    """
    api: paddle.einsum
    op version: 13
    """

    class Net(paddle.nn.Layer):
        """
        simple Net
        """

        def __init__(self):
            super(Net, self).__init__()

        def forward(self, x):
            """
            forward
            """
            x = paddle.einsum("ijk->kji", x)
            return x

    op = Net()
    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, "einsum_dot", [13])
    input_x = paddle.rand([2, 3, 2])
    obj.set_input_data("input_data", input_x)
    obj.run()


@_test_with_pir
def test_einsum_batch_matrix_multiplication():
    """
    api: paddle.einsum
    op version: 13
    """

    class Net(paddle.nn.Layer):
        """
        simple Net
        """

        def __init__(self):
            super(Net, self).__init__()

        def forward(self, x, y):
            """
            forward
            """
            x = paddle.einsum("ijk, ikl->ijl", x, y)
            return x

    op = Net()
    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, "einsum_dot", [13])
    input_x = paddle.rand([2, 3, 2])
    input_y = paddle.rand([2, 2, 3])
    obj.set_input_data("input_data", input_x, input_y)
    obj.run()


@_test_with_pir
def test_einsum_ellipsis_transpose():
    """
    api: paddle.einsum
    op version: 13
    """

    class Net(paddle.nn.Layer):
        """
        simple Net
        """

        def __init__(self):
            super(Net, self).__init__()

        def forward(self, x):
            """
            forward
            """
            x = paddle.einsum("...jk->...kj", x)
            return x

    op = Net()
    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, "einsum_dot", [13])
    input_x = paddle.rand([2, 3, 2])
    obj.set_input_data("input_data", input_x)
    obj.run()


@_test_with_pir
def test_einsum_ellipsis_batch_matrix_multiplication():
    """
    api: paddle.einsum
    op version: 13
    """

    class Net(paddle.nn.Layer):
        """
        simple Net
        """

        def __init__(self):
            super(Net, self).__init__()

        def forward(self, x, y):
            """
            forward
            """
            x = paddle.einsum("...jk, ...kl->...jl", x, y)
            return x

    op = Net()
    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, "einsum_dot", [13])
    input_x = paddle.rand([2, 3, 2])
    input_y = paddle.rand([2, 2, 3])
    obj.set_input_data("input_data", input_x, input_y)
    obj.run()


if __name__ == "__main__":
    test_einsum_sum()
    test_einsum_dot()
    test_einsum_outer()
    test_einsum_batch_matrix_multiplication()
    test_einsum_ellipsis_transpose()
    test_einsum_ellipsis_batch_matrix_multiplication()
