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
from onnxbase import _test_only_pir


class Net(paddle.nn.Layer):
    """
    simple Net
    """

    def __init__(self, config):
        super(Net, self).__init__()
        self.config = config

    def forward(self, input):
        """
        forward
        """
        return paddle._C_ops.set_value(
            input,
            self.config["starts"],
            self.config["ends"],
            self.config["steps"],
            self.config["axes"],
            self.config["decrease_axes"],
            self.config["none_axes"],
            self.config["shape"],
            self.config["values"],
        )


@_test_only_pir
def test_set_value_1():
    config = {
        "starts": [1, 3],
        "ends": [4, 4],
        "steps": [3, 1],
        "axes": [0, 1],
        "shape": [1, 1],
        "decrease_axes": [],
        "none_axes": [],
        "values": [2],
    }
    op = Net(config)
    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, "set_value", [19])
    obj.set_input_data("input_data", (paddle.ones(shape=[8, 8])))
    obj.run()


@_test_only_pir
def test_set_value_2():
    config = {
        "starts": [-3, 0],
        "ends": [100, -7],
        "steps": [1, 1],
        "axes": [1, 2],
        "decrease_axes": [],
        "none_axes": [],
        "shape": [1],
        "values": [6],
    }
    op = Net(config)
    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, "set_value", [19])
    obj.set_input_data("input_data", (paddle.ones(shape=[1, 7, 7, 1])))
    obj.run()
