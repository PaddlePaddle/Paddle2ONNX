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


class BaseNet(paddle.nn.Layer):
    def __init__(self):
        super(BaseNet, self).__init__()

    def forward(self, input):
        arr = paddle.tensor.create_array(dtype="float32")
        x = paddle.full(shape=[1, 3], fill_value=5, dtype="float32")
        i = paddle.zeros(shape=[1], dtype="int64")
        paddle.tensor.array_write(x, i, array=arr)
        item = paddle.tensor.array_read(arr, i)
        new_arr = paddle.tensor.create_array(dtype="float32")
        j = paddle.to_tensor([paddle.tensor.array_length(new_arr)], dtype="int64")
        paddle.tensor.array_write(item, j, array=new_arr)
        new_item = paddle.tensor.array_read(new_arr, j)
        return input + new_item


@_test_only_pir
def test_tensor_array():
    op = BaseNet()
    op.eval()
    obj = APIOnnx(op, "tensor_array", [11])
    obj.set_input_data("input_data", (paddle.rand([1, 3], dtype="float32")))
    obj.run()


if __name__ == "__main__":
    test_tensor_array()
