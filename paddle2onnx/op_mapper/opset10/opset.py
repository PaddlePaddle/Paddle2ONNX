#   Copyright (c) 2019  PaddlePaddle Authors. All Rights Reserved.
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

import math
import sys
import paddle2onnx
import os
import numpy as np
import paddle.fluid.core as core
import paddle.fluid as fluid
import onnx
from onnx import helper, onnx_pb
from paddle2onnx.op_mapper.opset9.opset import OpSet9


class OpSet10(OpSet9):
    def __init__(self):
        super(OpSet10, self).__init__()

    def slice(self, op, block):
        axes = op.attr('axes')
        starts = op.attr('starts')
        ends = op.attr('ends')
        axes_name = self.get_name(op.type, 'axes')
        starts_name = self.get_name(op.type, 'starts')
        ends_name = self.get_name(op.type, 'ends')

        axes_node = self.make_constant_node(axes_name,
                                            onnx_pb.TensorProto.INT64, axes)
        starts_node = self.make_constant_node(starts_name,
                                              onnx_pb.TensorProto.INT64, starts)
        ends_node = self.make_constant_node(ends_name,
                                            onnx_pb.TensorProto.INT64, ends)
        node = helper.make_node(
            "Slice",
            inputs=[op.input('Input')[0], starts_name, ends_name, axes_name],
            outputs=op.output('Out'), )
        return [starts_node, ends_node, axes_node, node]
