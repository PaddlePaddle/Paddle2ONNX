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
import paddle.nn.functional as F

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
        x = F.unfold(x, [3, 3], 1, 1, 1)
        return x
    

def test_unfold_11():
    """
    api: paddle.unfold
    op version: 11
    """
    op = Net()
    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, 'unfold', [11])
    obj.set_input_data(
        "input_data",
        paddle.to_tensor(
            randtool("float", -1, 1, [2,3,16,16]).astype('float32')))
    obj.run()


def test_unfold_11_2():
    """
    api: paddle.unfold
    op version: 11
    """
    op = Net()
    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, 'unfold', [11])
    obj.set_input_data(
        "input_data",
        paddle.arange(16).view([1,1,4, 4]).cast(paddle.float32)
    )
    obj.run()
if __name__ == "__main__":
    test_unfold_11()