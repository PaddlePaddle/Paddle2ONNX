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
    support_opset_version_range = (12, )

    @classmethod
    def opset_12(cls, graph, node, **kw):
        onnx_node = graph.make_node(
            'GreaterOrEqual',
            inputs=[node.input('X', 0), node.input('Y', 0)],
            outputs=node.output('Out'))


@op_mapper('equal')
class Equal():
    support_opset_version_range = (1, 12)

    @classmethod
    def opset_1(cls, graph, node, **kw):
        onnx_node = graph.make_node(
            'Equal',
            inputs=[node.input('X', 0), node.input('Y', 0)],
            outputs=node.output('Out'))


@op_mapper('not_equal')
class NotEqual():
    support_opset_version_range = (12, )

    @classmethod
    def opset_1(cls, graph, node, **kw):
        equal_val = graph.make_node(
            'Equal', inputs=[node.input('X', 0), node.input('Y', 0)])
        k_node = graph.make_node(
            'Cast', inputs=[equal_val], to=dtypes.ONNX.INT64)
        const = graph.make_node('Constant', dtype=dtypes.ONNX.INT64, value=1)
        sub_ = graph.make_node('Sub', inputs=[const, k_node])
        graph.make_node(
            'Cast',
            inputs=[sub_],
            outputs=node.output('Out'),
            to=dtypes.ONNX.BOOL)


@op_mapper('greater_than')
class GreaterThan():
    support_opset_version_range = (1, )

    @classmethod
    def opset_1(cls, graph, node, **kw):
        onnx_node = graph.make_node(
            'Greater',
            inputs=[node.input('X', 0), node.input('Y', 0)],
            outputs=node.output('Out'))


@op_mapper('logical_and')
class LogicalAnd():
    support_opset_version_range = (1, )

    @classmethod
    def opset_1(cls, graph, node, **kw):
        onnx_node = graph.make_node(
            'And',
            inputs=[node.input('X', 0), node.input('Y', 0)],
            outputs=node.output('Out'))


@op_mapper('logical_not')
class LogicalNot():
    support_opset_version_range = (1, 12)

    @classmethod
    def opset_1(cls, graph, node, **kw):
        graph.make_node(
            'Not', inputs=node.input('X'), outputs=node.output('Out'))


@op_mapper('logical_or')
class LogicalOr():
    support_opset_version_range = (7, 12)

    @classmethod
    def opset_7(cls, graph, node, **kw):
        graph.make_node(
            'Or',
            inputs=[node.input('X', 0), node.input('Y', 0)],
            outputs=node.output('Out'))


@op_mapper('logical_xor')
class LogicalXOr():
    support_opset_version_range = (7, 12)

    @classmethod
    def opset_7(cls, graph, node, **kw):
        graph.make_node(
            'Xor',
            inputs=[node.input('X', 0), node.input('Y', 0)],
            outputs=node.output('Out'))


@op_mapper('less_equal')
class LessOrEqual():
    support_opset_version_range = (12, )

    @classmethod
    def opset_12(cls, graph, node, **kw):
        onnx_node = graph.make_node(
            'LessOrEqual',
            inputs=[node.input('X', 0), node.input('Y', 0)],
            outputs=node.output('Out'))


@op_mapper('less_than')
class Less_than():
    support_opset_version_range = (7, 12)

    @classmethod
    def opset_7(cls, graph, node, **kw):
        graph.make_node(
            'Less',
            inputs=[node.input('X', 0), node.input('Y', 0)],
            outputs=node.output('Out'), )


@op_mapper('isfinite_v2')
class Isfinite():
    support_opset_version_range = (10, 12)

    @classmethod
    def opset_10(cls, graph, node, **kw):
        is_inf = graph.make_node('IsInf', inputs=node.input('X', 0))
        is_nan = graph.make_node('IsNaN', inputs=node.input('X', 0))
        finite = graph.make_node('Or', inputs=[is_inf, is_nan])
        graph.make_node('Not', inputs=[finite], outputs=node.output('Out'))


@op_mapper('isinf_v2')
class IsInf():
    support_opset_version_range = (10, 12)

    @classmethod
    def opset_10(cls, graph, node, **kw):
        graph.make_node(
            'IsInf', inputs=node.input('X'), outputs=node.output('Out'))


@op_mapper('isnan_v2')
class IsNaN():
    support_opset_version_range = (9, 12)

    @classmethod
    def opset_9(cls, graph, node, **kw):
        graph.make_node(
            'IsNaN', inputs=node.input('X'), outputs=node.output('Out'))


@op_mapper('isnan')
class IsNaN():
    support_opset_version_range = (9, 12)

    @classmethod
    def opset_9(cls, graph, node, **kw):
        isnan = graph.make_node('IsNaN', inputs=node.input('X'))
        cast_node = graph.make_node(
            'Cast', inputs=isnan, attrs={'to': dtypes.ONNX.FLOAT})
        reduce_node = graph.make_node(
            'ReduceMax', inputs=[cast_node], keepdims=False)
        graph.make_node(
            'Unsqueeze',
            inputs=[reduce_node],
            outputs=node.output('Out'),
            axes=[0])

    @classmethod
    def opset_13(cls, graph, node, **kw):
        isnan = graph.make_node('IsNaN', inputs=node.input('X'))
        cast_node = graph.make_node(
            'Cast', inputs=isnan, attrs={'to': dtypes.ONNX.FLOAT})
        reduce_node = graph.make_node(
            'ReduceMax', inputs=[cast_node], keepdims=False)

        axis = graph.make_node('Constant', dtype=dtypes.ONNX.INT64, value=[0])
        graph.make_node(
            'Unsqueeze', inputs=[reduce_node, axis], outputs=node.output('Out'))
