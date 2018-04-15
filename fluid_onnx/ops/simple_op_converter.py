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

import sys
import itertools
import onnx
from fluid_onnx.ops.utils import ConversionType, conversion_checker
from fluid_onnx.ops.utils import parse_fluid_op


class SimpleOpConverter(object):
    def __init__(self,
                 source_op_type,
                 target_op_type,
                 convert_type=ConversionType.FLUID_OP_TO_ONNX_NODE,
                 **target_attrs):
        self._source_op_type = source_op_type
        self._target_op_type = target_op_type
        self._convert_type = convert_type
        self._target_attrs = target_attrs

    @property
    def source(self):
        return self._source_op_type

    @property
    def target(self):
        return self._target_op_type

    @property
    def convert_type(self):
        return self._convert_type

    @conversion_checker
    def __call__(self, operator, **kwargs):
        inputs, outputs, attrs = parse_fluid_op(operator)
        # @TODO Currently, only consider operator containing inputs and outputs
        # May support common attributes conversion
        kwargs.update(self._target_attrs)
        flatted_inputs = list(itertools.chain(*map(lambda x: x[1], inputs)))
        flatted_outputs = list(itertools.chain(*map(lambda x: x[1], outputs)))
        onnx_node = onnx.helper.make_node(
            self.target,
            inputs=flatted_inputs,
            outputs=flatted_outputs,
            **kwargs)
        return onnx_node
