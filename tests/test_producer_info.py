# Copyright (c) 2026  PaddlePaddle Authors. All Rights Reserved.
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

import importlib.metadata
import tempfile
from pathlib import Path

import onnx
import paddle
import paddle2onnx
from onnxbase import _test_with_pir


class Net(paddle.nn.Layer):
    def __init__(self):
        super(Net, self).__init__()
        self.fc = paddle.nn.Linear(4, 2)

    def forward(self, x):
        return self.fc(x)


@_test_with_pir
def test_producer_info():
    """The exported ModelProto records paddle2onnx as producer, with its version."""
    net = Net()
    net.eval()
    with tempfile.TemporaryDirectory() as tmp:
        prefix = str(Path(tmp) / "model")
        paddle.jit.save(
            net, prefix, [paddle.static.InputSpec([None, 4], "float32", "x")]
        )
        pir = paddle.get_flags("FLAGS_enable_pir_api")["FLAGS_enable_pir_api"]
        model_file = prefix + (".json" if pir else ".pdmodel")
        onnx_bytes = paddle2onnx.export(model_file, prefix + ".pdiparams")
    model = onnx.load_from_string(onnx_bytes)
    assert model.producer_name == "paddle2onnx"
    assert model.producer_version == importlib.metadata.version("paddle2onnx")
