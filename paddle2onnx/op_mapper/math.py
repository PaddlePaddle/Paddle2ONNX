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


@op_mapper('matmul')
class MatMul():
    support_opset_verision_range = (1, 12)

    @classmethod
    def opset_1(cls, graph, node, **kw):
        x = node.input('X', idx=0)
        y = node.input('Y', idx=0)
        graph.make_node('MatMul', inputs=[x, y], outputs=node.output('Out'))


@op_mapper('exp')
class Exp():
    support_opset_verision_range = (1, 12)

    @classmethod
    def opset_1(cls, graph, node, **kw):
        graph.make_node(
            'Exp', inputs=node.input('X'), outputs=node.output('Out'))


@op_mapper('abs')
class Abs:
    support_opset_verision_range = (1, 12)

    @classmethod
    def opset_1(cls, graph, node, **kw):
        graph.make_node(
            'Abs', inputs=node.input('X'), outputs=node.output('Out'))


@op_mapper(
    [
        'elementwise_add', 'elementwise_sub', 'elementwise_div',
        'elementwise_mul'
    ],
    mapper_dict={
        'elementwise_add': 'Add',
        'elementwise_sub': 'Sub',
        'elementwise_div': 'Div',
        'elementwise_mul': 'Mul',
    })
class ElementwiseOps():
    support_opset_verision_range = (7, 12)

    @classmethod
    def opset_7(cls, graph, node, **kw):
        op_type = kw['mapper_dict'][node.type]
        axis = node.attr('axis')
        x = node.input('X', 0)
        y = node.input('Y', 0)
        x_shape = node.input_shape('X', 0)
        y_shape = node.input_shape('Y', 0)
        if axis == -1 or axis == (len(x_shape) - 1
                                  ) or len(x_shape) == len(y_shape):
            onnx_node = graph.make_node(
                op_type, inputs=[x, y], outputs=node.output('Out'))
        else:
            broadcast_shape = [1] * len(x_shape)
            broadcast_shape[axis:axis + len(y_shape)] = y_shape
            broadcast_shape_node = graph.make_node(
                'Constant',
                dtype=dtypes.ONNX.INT64,
                value=list(broadcast_shape))
            y_node = graph.make_node(
                'Reshape', inputs=[y, broadcast_shape_node])
            onnx_node = graph.make_node(
                op_type, inputs=[x, y_node], outputs=node.output('Out'))


@op_mapper('pow')
class Pow():
    support_opset_verision_range = (8, 12)

    @classmethod
    def opset_8(cls, graph, node, **kw):
        x = node.input('X', 0)
        factor = node.attr('factor')
        factor_node = graph.make_node(
            'Constant',
            inputs=[],
            dims=[1],
            dtype=dtypes.ONNX.FLOAT,
            value=factor)
        x_shape = graph.make_node('Shape', inputs=[x])
        factor_broadcast = graph.make_node(
            'Expand', inputs=[factor_node, x_shape])
        onnx_node = graph.make_node(
            'Pow', inputs=[x, factor_broadcast], outputs=node.output('Out'))


@op_mapper('mul')
class Mul():
    support_opset_verision_range = (1, 12)

    @classmethod
    def opset_1(cls, graph, node, **kw):
        x = node.input('X', 0)
        y = node.input('Y', 0)
        out = node.output('Out', 0)
        x_shape = node.input_shape('X', 0)
        y_shape = node.input_shape('Y', 0)
        x_num_col_dims = node.attr('x_num_col_dims')
        y_num_col_dims = node.attr('y_num_col_dims')
        flatten_x = graph.make_node(
            'Flatten', inputs=node.input('X'), attrs={'axis': x_num_col_dims})
        flatten_y = graph.make_node(
            'Flatten', inputs=node.input('Y'), attrs={'axis': y_num_col_dims})
        mul_node = graph.make_node(
            'MatMul', inputs=[flatten_x, flatten_y], outputs=node.output('Out'))


@op_mapper('sum')
class Sum():
    support_opset_verison_range = (1, 12)

    def opset_1(cls, graph, node, **kw):
        graph.make_node(
            'sum', inputs=node.input('X'), outputs=node.output('Out'))


@op_mapper('floor')
class Floor():
    support_opset_verison_range = (1, 12)

    @classmethod
    def opset_1(cls, graph, node, **kw):
        graph.make_node(
            'Floor', inputs=node.input('X'), outputs=node.output('Out'))


@op_mapper(
    ['reduce_mean', 'reduce_sum', 'reduce_min', 'reduce_max'],
    mapper_dict={
        'reduce_mean': 'ReduceMean',
        'reduce_sum': 'ReduceSum',
        'reduce_min': 'ReduceMin',
        'reduce_max': 'ReudceMax'
    })
class ReduceMean():
    support_opset_verison_range = (1, 12)

    @classmethod
    def opset_1(cls, graph, node, **kw):
        op_type = kw['mapper_dict'][node.type]
        graph.make_node(
            op_type,
            inputs=node.input('X'),
            outputs=node.output('Out'),
            attrs={
                'axes': node.attr('dim'),
                'keepdims': node.attr('keep_dim')
            })


@op_mapper('arg_max')
class ArgMax():
    support_opset_verison_range = (1, 12)

    @classmethod
    def opset_1(cls, graph, node, **kw):
        graph.make_node(
            'ArgMax',
            inputs=node.input('X'),
            outputs=node.output('Out'),
            attrs={'axis': node.attr('axis'),
                   'keepdims': 0})


@op_mapper('scale')
class Scale():
    support_opset_verison_range = (1, 12)

    @classmethod
    def opset_1(cls, graph, node, **kw):
        scale = node.attr('scale')
        bias = node.attr('bias')
        if np.fabs(scale - 1.0) < 1e-06 and np.fabs(bias - 0.0) < 1e-06:
            graph.make_node(
                'Identity', inputs=node.input('X'), outputs=node.output('Out'))
        else:
            raise Exception(
                "please try to convert OP:scale with opset_version >= 7.")

    @classmethod
    def opset_7(cls, graph, node, **kw):
        scale = node.attr('scale')
        bias = node.attr('bias')
        if np.fabs(scale - 1.0) < 1e-06 and np.fabs(bias - 0.0) < 1e-06:
            graph.make_node(
                'Identity', inputs=node.input('X'), outputs=node.output('Out'))
        else:
            scale_node = graph.make_node(
                'Constant', attrs={'dtype': dtypes.ONNX.FLOAT,
                                   'value': scale})
            bias_node = graph.make_node(
                'Constant', attrs={'dtype': dtypes.ONNX.FLOAT,
                                   'value': bias})
            cast_node = graph.make_node(
                'Cast', inputs=node.input('X'),
                attrs={'to': dtypes.ONNX.FLOAT})
            if node.attr('bias_after_scale'):
                node1 = graph.make_node('Mul', inputs=[scale_node, cast_node])
                node2 = graph.make_node(
                    'Add',
                    inputs=[bias_node, node1],
                    outputs=node.output('Out'))
            else:
                node1 = graph.make_node('Add', inputs=[bias_node, cast_node])
                node2 = graph.make_node(
                    'Mul',
                    inputs=[scale_node, node1],
                    outputs=[node.output('Out')])


@op_mapper('softmax')
class Softmax():
    support_opset_verison_range = (1, 12)

    @classmethod
    def opset_1(cls, graph, node, **kw):
        axis = node.attr('axis')
        shape = node.output_shape('Out', 0)
        if axis < 0:
            axis += len(shape)
        if axis == len(shape) - 1:
            node = graph.make_node(
                'Softmax',
                inputs=node.input('X'),
                outputs=node.output('Out'),
                attrs={'axis': node.attr('axis')})
        else:
            perm = [i for i in range(len(shape))]
            perm[-1] = axis
            perm[axis] = len(shape) - 1
            transpose_node = graph.make_node(
                'Transpose', inputs=node.input('X'), attrs={'perm': perm})
            softmax_node = graph.make_node(
                'Softmax', inputs=[transpose_node], axis=-1)
            transpose_node1 = graph.make_node(
                'Transpose',
                inputs=[softmax_node],
                outputs=node.output('Out'),
                attrs={'perm': perm})
