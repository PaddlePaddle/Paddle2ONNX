# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from fluid_onnx.ops.simple_op_converter import SimpleOpConverter

_fluid_op_convert_mapper = {}


def get_fluid_op_converter(op_type):
    if op_type not in _fluid_op_convert_mapper:
        raise Exception("Not supported conversion for fluid op %s." % op_type)
    return _fluid_op_convert_mapper[op_type]


def _register_op(op_type, converter):
    _fluid_op_convert_mapper[op_type] = converter


_register_op("mul", SimpleOpConverter("mul", "MatMul"))
_register_op(
    "elementwise_add", SimpleOpConverter(
        "elementwise_add", "Add", broadcast=1))
