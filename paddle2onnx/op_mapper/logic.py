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

import numpy as np
from paddle2onnx.constant import dtypes
from paddle2onnx.op_mapper import OpMapper as op_mapper


@op_mapper('greater_equal')
class GreaterOrEqual():
    support_opset_verison_range = (12, )

    @classmethod
    def opset_12(cls, graph, node, **kw):
        onnx_node = graph.make_node(
            'GreaterOrEqual',
            inputs=[node.input('X', 0), node.input('Y', 0)],
            outputs=node.output('Out'))


@op_mapper('equal')
class Equal():
    support_opset_verison_range = (12, )

    @classmethod
    def opset_1(cls, graph, node, **kw):
        onnx_node = graph.make_node(
            'Equal',
            inputs=[node.input('X', 0), node.input('Y', 0)],
            outputs=node.output('Out'))


@op_mapper('greater_than')
class GreaterThan():
    support_opset_verison_range = (1, )

    @classmethod
    def opset_1(cls, graph, node, **kw):
        onnx_node = graph.make_node(
            'Greater',
            inputs=[node.input('X', 0), node.input('Y', 0)],
            outputs=node.output('Out'))


@op_mapper('logical_and')
class LogicalAnd():
    support_opset_verison_range = (1, )

    @classmethod
    def opset_1(cls, graph, node, **kw):
        onnx_node = graph.make_node(
            'And',
            inputs=[node.input('X', 0), node.input('Y', 0)],
            outputs=node.output('Out'))


@op_mapper('less_equal')
class LessOrEqual():
    support_opset_verison_range = (12, )

    @classmethod
    def opset_12(cls, graph, node, **kw):
        onnx_node = graph.make_node(
            'LessOrEqual',
            inputs=[node.input('X', 0), node.input('Y', 0)],
            outputs=node.output('Out'))


@op_mapper('equal')
class Equal():
    support_opset_verison_range = (1, )

    @classmethod
    def opset_12(cls, graph, node, **kw):
        onnx_node = graph.make_node(
            'Equal',
            inputs=[node.input('X', 0), node.input('Y', 0)],
            outputs=node.output('Out'))
