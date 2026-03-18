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

global_config = {
    "axis": 0,
    "use_stack": False,
}


class BaseNet(paddle.nn.Layer):
    def __init__(self, axis, use_stack):
        super(BaseNet, self).__init__()
        self.axis = axis
        self.use_stack = use_stack

    def forward(self, x0, x1):
        i = paddle.full(shape=[1], dtype="int64", fill_value=0)
        array = paddle.tensor.array.create_array(dtype="float32")
        paddle.tensor.array.array_write(x0, i, array)
        paddle.tensor.array.array_write(x1, i + 1, array)
        output, output_index = paddle.tensor.manipulation.tensor_array_to_tensor(
            input=array, axis=self.axis, use_stack=self.use_stack
        )
        output_index = output_index.astype(
            "int64"
        )  # if not cast, the dtype of output_index is int32 in static graph but int64 in dynamic graph
        return output, output_index


@_test_only_pir
def test_array_to_tensor_1():
    global global_config
    op = BaseNet(0, False)
    op.eval()
    obj = APIOnnx(op, "array_to_tensor", [17])
    obj.set_input_data(
        "input_data",
        (paddle.rand([2, 2], dtype="float32"), paddle.rand([2, 2], dtype="float32")),
    )
    obj.run()


@_test_only_pir
def test_array_to_tensor_2():
    global global_config
    op = BaseNet(1, True)
    op.eval()
    obj = APIOnnx(op, "array_to_tensor", [17])
    obj.set_input_data(
        "input_data",
        (paddle.rand([2, 2], dtype="float32"), paddle.rand([2, 2], dtype="float32")),
    )
    obj.run()


if __name__ == "__main__":
    test_array_to_tensor_1()
    test_array_to_tensor_2()
