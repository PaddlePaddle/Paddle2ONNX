# Copyright (c) 2022  PaddlePaddle Authors. All Rights Reserved.
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

from paddle2onnx.utils import logging
from . import command
from .convert import dygraph2onnx
from .convert import program2onnx

def run_convert(model, input_shape_dict=None, scope=None, opset_version=9):
    logging.warning("[Deprecated] `paddle2onnx.run_convert` will be deprecated in the future version, the recommended usage is `paddle2onnx.export`")
    from paddle2onnx.legacy import run_convert
    return run_convert(model, input_shape_dict, scope, opset_version)

