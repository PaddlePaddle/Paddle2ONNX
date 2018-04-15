#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

import paddle.fluid as fluid
import traceback


class ConversionType:
    FLUID_OP_TO_ONNX_NODE = "Fluid operator to onnx node"


class ConversionError(Exception):
    def __init__(self, source, target, conversion_type, err):
        self.source = source
        self.target = target
        self.conversion_type = conversion_type
        self.err = err

    def __str__(self):
        return "%s conversion: %s --> %s failed. \n%s" % \
                (self.conversion_type, self.source, self.target, self.err)


def conversion_checker(convert):
    def checker_warpper(self, *args, **kwargs):
        try:
            return convert(self, *args, **kwargs)
        except:
            raise ConversionError(self.source, self.target, self.convert_type,
                                  traceback.format_exc())

    return checker_warpper


def parse_fluid_op(op):
    # keep order for inputs and outputs
    inputs = [(name, op.input(name)) for name in op.input_names]
    outputs = [(name, op.output(name)) for name in op.output_names]

    attrs = None
    if op.attr_names is not None:
        attrs = dict([(name, op.attr(name)) for name in op.attr_names])

    return inputs, outputs, attrs
