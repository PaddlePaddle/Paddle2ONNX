# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
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
import unittest
import paddle2onnx


class SimpleNet(paddle.nn.Layer):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.fc = paddle.nn.Linear(10, 10)

    def forward(self, x):
        return self.fc(x)


class TestDygraph2OnnxAPI(unittest.TestCase):
    def test_api(self):
        net = SimpleNet()
        input_spec = [paddle.static.InputSpec(shape=[None, 10], dtype="float32")]
        paddle2onnx.dygraph2onnx(net, "simple_net.onnx", input_spec)


if __name__ == "__main__":
    unittest.main()
