#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
from __future__ import absolute_import

__version__ = "0.4"

import paddle
v0, v1, v2 = paddle.__version__.split('.')
if v0 == '0' and v1 == '0' and v2 == '0':
    from .convert import convert_dygraph_to_onnx
elif int(v0) > 2:
    from .convert import convert_dygraph_to_onnx

from . import graph
from .op_mapper import onnx_opset
from .convert_program import convert_program_to_onnx
