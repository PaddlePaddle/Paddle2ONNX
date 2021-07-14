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
from paddle2onnx.op_mapper import mapper_helper


@op_mapper('matmul')
class MatMul():
    support_opset_verision_range = (1, 12)

    @classmethod
    def opset_1(cls, graph, node, **kw):
        x = node.input('X', idx=0)
        y = node.input('Y', idx=0)
        if node.attr('transpose_X'):
            perm = list(range(len(node.input_shape('X', 0))))
            perm[-1], perm[-2] = perm[-2], perm[-1]
            x = graph.make_node('Transpose', inputs=[x], perm=perm)
        if node.attr('transpose_Y'):
            perm = list(range(len(node.input_shape('Y', 0))))
            perm[-1], perm[-2] = perm[-2], perm[-1]
            y = graph.make_node('Transpose', inputs=[y], perm=perm)
        if node.attr('alpha') == 1.0:
            graph.make_node('MatMul', inputs=[x, y], outputs=node.output('Out'))
        else:
            matmul = graph.make_node('MatMul', inputs=[x, y])
            scale = graph.make_node(
                'Constant', dtype=dtypes.ONNX.FLOAT, value=node.attr('alpha'))
            onnx_node = graph.make_node(
                'Mul', inputs=[matmul, scale], outputs=node.output('Out'))


@op_mapper('matmul_v2')
class MatMul():
    support_opset_verision_range = (1, 12)

    @classmethod
    def opset_1(cls, graph, node, **kw):
        x = node.input('X', idx=0)
        y = node.input('Y', idx=0)
        out = node.output('Out')
        if node.attr('trans_x'):
            perm = list(range(len(node.input_shape('X', 0))))
            perm[-1], perm[-2] = perm[-2], perm[-1]
            x = graph.make_node('Transpose', inputs=[x], perm=perm)
        if node.attr('trans_y'):
            perm = list(range(len(node.input_shape('Y', 0))))
            perm[-1], perm[-2] = perm[-2], perm[-1]
            y = graph.make_node('Transpose', inputs=[y], perm=perm)
        graph.make_node('MatMul', inputs=[x, y], outputs=out)


@op_mapper('exp')
class Exp():
    support_opset_version_range = (1, 12)

    @classmethod
    def opset_1(cls, graph, node, **kw):
        graph.make_node(
            'Exp', inputs=node.input('X'), outputs=node.output('Out'))


@op_mapper('abs')
class Abs:
    support_opset_version_range = (1, 12)

    @classmethod
    def opset_1(cls, graph, node, **kw):
        graph.make_node(
            'Abs', inputs=node.input('X'), outputs=node.output('Out'))


@op_mapper(
    [
        'elementwise_add',
        'elementwise_sub',
        'elementwise_div',
        'elementwise_mul',
        'elementwise_min',
        'elementwise_max',
        'elementwise_pow',
        'elementwise_mod',
    ],
    mapper_dict={
        'elementwise_add': 'Add',
        'elementwise_sub': 'Sub',
        'elementwise_div': 'Div',
        'elementwise_mul': 'Mul',
        'elementwise_min': 'Min',
        'elementwise_max': 'Max',
        'elementwise_pow': 'Pow',
        'elementwise_mod': 'Mod',
    })
class ElementwiseOps():
    support_opset_version_range = (7, 12)

    @classmethod
    def opset_9(cls, graph, node, **kw):
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


@op_mapper('elementwise_floordiv')
class ElementWiseFloorDiv():
    support_opset_version_range = (11, 12)

    @classmethod
    def opset_7(cls, graph, node, **kw):
        x = node.input('X', 0)
        y = node.input('Y', 0)
        axis = node.attr('axis')
        x_shape = node.input_shape('X', 0)
        y_shape = node.input_shape('Y', 0)
        x_dtype = node.input_dtype('X', 0)
        y_dtype = node.input_dtype('Y', 0)
        x_dtype = dtypes.DTYPE_PADDLE_STR_MAP[x_dtype]
        y_dtype = dtypes.DTYPE_PADDLE_STR_MAP[y_dtype]
        is_int = False
        if x_dtype.count('int') > 0 and y_dtype.count('int') > 0:
            is_int = True
        if axis == -1 or axis == (len(x_shape) - 1
                                  ) or len(x_shape) == len(y_shape):
            if is_int:
                graph.make_node(
                    'Div', inputs=[x, y], outputs=node.output('Out'))
            else:
                div_node = graph.make_node('Div', inputs=[x, y])
                graph.make_node(
                    'Floor', inputs=[div_node], outputs=node.output('Out'))
        else:
            broadcast_shape = [1] * len(x_shape)
            broadcast_shape[axis:axis + len(y_shape)] = y_shape
            broadcast_shape_node = graph.make_node(
                'Constant',
                dtype=dtypes.ONNX.INT64,
                value=list(broadcast_shape))
            y_node = graph.make_node(
                'Reshape', inputs=[y, broadcast_shape_node])
            if is_int:
                div_node = graph.make_node(
                    'Div', inputs=[x, y_node], outputs=node.output('Out'))
            else:
                div_node = graph.make_node('Div', inputs=[x, y_node])
                graph.make_node(
                    'Floor', inputs=[div_node], outputs=node.output('Out'))


@op_mapper('pow')
class Pow():
    support_opset_version_range = (8, 12)

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


@op_mapper('square')
class Square():
    support_opset_verision_range = (7, 12)

    @classmethod
    def opset_7(cls, graph, node, **kw):
        x = node.input('X', 0)
        onnx_node = graph.make_node(
            'Mul', inputs=[x, x], outputs=node.output('Out'))


@op_mapper('cumsum')
class CumSum():
    support_opset_version_range = (11, 12)

    @classmethod
    def opset_11(cls, graph, node, **kw):

        axis = graph.make_node(
            'Constant', dtype=dtypes.ONNX.INT64, value=node.attr('axis'))
        graph.make_node(
            'CumSum',
            inputs=[node.input('X', 0), axis],
            outputs=node.output('Out'))


@op_mapper('mul')
class Mul():
    support_opset_version_range = (1, 12)

    @classmethod
    def opset_1(cls, graph, node, **kw):
        x = node.input('X', 0)
        y = node.input('Y', 0)
        out = node.output('Out', 0)
        x_num_col_dims = node.attr('x_num_col_dims')
        y_num_col_dims = node.attr('y_num_col_dims')
        flatten_x = graph.make_node(
            'Flatten', inputs=node.input('X'), attrs={'axis': x_num_col_dims})
        flatten_y = graph.make_node(
            'Flatten', inputs=node.input('Y'), attrs={'axis': y_num_col_dims})
        mul_node = graph.make_node('MatMul', inputs=[flatten_x, flatten_y])

        x_shape = graph.make_node('Shape', inputs=[x])
        l_shape = mapper_helper.slice_helper(
            graph, x_shape, axes=[0], starts=[0], ends=[x_num_col_dims])
        y_shape = graph.make_node('Shape', inputs=[y])
        y_rank = len(node.input_shape('Y', 0))
        r_shape = mapper_helper.slice_helper(
            graph, y_shape, axes=[0], starts=[y_num_col_dims], ends=[y_rank])

        out_shape = graph.make_node('Concat', inputs=[l_shape, r_shape], axis=0)
        graph.make_node('Reshape', [mul_node, out_shape], node.output('Out'))


@op_mapper('affine_channel')
class AffineChannel():
    support_opset_version_range = (1, 12)

    @classmethod
    def opset_1(cls, graph, node, **kw):
        x = node.input('X', 0)
        bias = node.input('Bias', 0)
        scale = node.input('Scale', 0)
        scale = graph.make_node('Unsqueeze', inputs=scale, axes=[0, 2, 3])
        bias = graph.make_node('Unsqueeze', inputs=bias, axes=[0, 2, 3])
        x = graph.make_node('Mul', inputs=[x, scale])
        x = graph.make_node('Add', inputs=[x, bias], outputs=node.output('Out'))


@op_mapper('bmm')
class BMM():
    support_opset_version_range = (1, 12)

    @classmethod
    def opset_1(cls, graph, node, **kw):
        x = node.input('X', 0)
        y = node.input('Y', 0)
        mul_node = graph.make_node(
            'MatMul', inputs=[x, y], outputs=node.output('Out'))


@op_mapper('p_norm')
class PNorm():
    support_opset_version_range = (1, 12)

    @classmethod
    def opset_1(cls, graph, node, **kw):
        x = node.input('X', 0)
        axis = node.attr('axis')
        p = node.attr('porder')
        keepdim = node.attr('keepdim')
        epsilon = node.attr('epsilon')
        assert axis == 1, "Only axis == 1 is supported for p_norm"
        if p == 1 or p == 2 and not keepdim:
            graph.make_node(
                'LpNormalization',
                inputs=[x],
                outputs=node.output('Out'),
                axis=1,
                p=p)
        else:
            pnode = graph.make_node(
                'Constant', dtype=dtypes.ONNX.FLOAT, value=[p])
            mul = graph.make_node('Pow', inputs=[x, pnode])
            reduce_sum = graph.make_node(
                'ReduceSum', inputs=[mul], axes=[1], keepdims=keepdim)
            pnode1 = graph.make_node(
                'Constant', dtype=dtypes.ONNX.FLOAT, value=[1.0 / p])
            graph.make_node(
                'Pow', inputs=[reduce_sum, pnode1], outputs=node.output('Out'))

    @classmethod
    def opset_13(cls, graph, node, **kw):
        x = node.input('X', 0)
        axis = node.attr('axis')
        p = node.attr('porder')
        keepdim = node.attr('keepdim')
        epsilon = node.attr('epsilon')
        assert axis == 1, "Only axis == 1 is supported for p_norm"
        if (p == 1 or p == 2) and not keepdim:
            graph.make_node(
                'LpNormalization',
                inputs=[x],
                outputs=node.output('Out'),
                axis=1,
                p=p)
        else:
            pnode = graph.make_node(
                'Constant', dtype=dtypes.ONNX.FLOAT, value=[p])
            mul = graph.make_node('Pow', inputs=[x, pnode])
            axes = graph.make_node(
                'Constant', dtype=dtypes.ONNX.INT64, value=[1])
            reduce_sum = graph.make_node(
                'ReduceSum', inputs=[mul, axes], keepdims=keepdim)
            pnode1 = graph.make_node(
                'Constant', dtype=dtypes.ONNX.FLOAT, value=[1.0 / p])
            graph.make_node(
                'Pow', inputs=[reduce_sum, pnode1], outputs=node.output('Out'))


@op_mapper('sum')
class Sum():
    support_opset_version_range = (1, 12)

    def opset_1(cls, graph, node, **kw):
        graph.make_node(
            'sum', inputs=node.input('X'), outputs=node.output('Out'))


@op_mapper('floor')
class Floor():
    support_opset_version_range = (1, 12)

    @classmethod
    def opset_1(cls, graph, node, **kw):
        graph.make_node(
            'Floor', inputs=node.input('X'), outputs=node.output('Out'))


@op_mapper(
    ['reduce_mean', 'reduce_sum', 'reduce_min', 'reduce_max', 'reduce_prod'],
    mapper_dict={
        'reduce_mean': 'ReduceMean',
        'reduce_sum': 'ReduceSum',
        'reduce_min': 'ReduceMin',
        'reduce_max': 'ReduceMax',
        'reduce_prod': 'ReduceProd'
    })
class ReduceMean():
    support_opset_version_range = (1, 12)

    @classmethod
    def opset_1(cls, graph, node, **kw):
        op_type = kw['mapper_dict'][node.type]

        output_shape = node.output_shape('Out', 0)
        need_unsqueeze = False
        if not node.attr('keep_dim'):
            if list(output_shape) == [1]:
                need_unsqueeze = True

        if not need_unsqueeze:
            graph.make_node(
                op_type,
                inputs=node.input('X'),
                outputs=node.output('Out'),
                attrs={
                    'axes': node.attr('dim'),
                    'keepdims': node.attr('keep_dim')
                })
        else:
            reduce_node = graph.make_node(
                op_type,
                inputs=node.input('X'),
                attrs={
                    'axes': node.attr('dim'),
                    'keepdims': node.attr('keep_dim')
                })
            graph.make_node(
                'Unsqueeze',
                inputs=[reduce_node],
                outputs=node.output('Out'),
                axes=[0])


@op_mapper('mean')
class Mean():
    support_opset_verison_range = (1, 12)

    @classmethod
    def opset_1(cls, graph, node, **kw):
        graph.make_node(
            'ReduceMean',
            inputs=node.input('X'),
            outputs=node.output('Out'),
            keepdims=0)


@op_mapper('arg_max')
class ArgMax():
    support_opset_version_range = (1, 12)

    @classmethod
    def opset_1(cls, graph, node, **kw):
        graph.make_node(
            'ArgMax',
            inputs=node.input('X'),
            outputs=node.output('Out'),
            attrs={'axis': node.attr('axis'),
                   'keepdims': 0})


#
#@op_mapper('scale')
#class Scale():
#    support_opset_version_range = (1, 12)
#
#    @classmethod
#    def opset_1(cls, graph, node, **kw):
#        scale = node.attr('scale')
#        bias = node.attr('bias')
#        if np.fabs(scale - 1.0) < 1e-06 and np.fabs(bias - 0.0) < 1e-06:
#            graph.make_node(
#                'Identity', inputs=node.input('X'), outputs=node.output('Out'))
#        else:
#            raise Exception(
#                "please try to convert OP:scale with opset_version >= 7.")
#
#    @classmethod
#    def opset_7(cls, graph, node, **kw):
#        scale = node.attr('scale')
#        bias = node.attr('bias')
#        if np.fabs(scale - 1.0) < 1e-06 and np.fabs(bias - 0.0) < 1e-06:
#            graph.make_node(
#                'Identity', inputs=node.input('X'), outputs=node.output('Out'))
#        else:
#            cast_node = graph.make_node(
#                'Cast', inputs=node.input('X'),
#                attrs={'to': dtypes.ONNX.FLOAT})
#            if np.fabs(scale - 1.0) < 1e-06:
#                bias_node = graph.make_node(
#                    'Constant',
#                    attrs={'dtype': dtypes.ONNX.FLOAT,
#                           'value': [bias]})
#                graph.make_node('Add', inputs=[cast_node, bias_node], outputs=node.output('Out'))
#            elif np.fabs(bias - 1.0) < 1e-06:
#                scale_node = graph.make_node(
#                   'Constant',
#                   attrs={'dtype': dtypes.ONNX.FLOAT,
#                          'value': [scale]})
#                graph.make_node('Mul', inputs=[cast_node, scale_node], outputs=node.output('Out'))
#            else:
#                scale_node = graph.make_node(
#                    'Constant',
#                    attrs={'dtype': dtypes.ONNX.FLOAT,
#                           'value': [scale]})
#                bias_node = graph.make_node(
#                    'Constant',
#                    attrs={'dtype': dtypes.ONNX.FLOAT,
#                           'value': [bias]})
#                if node.attr('bias_after_scale'):
#                    node1 = graph.make_node('Mul', inputs=[cast_node, scale_node])
#                    node2 = graph.make_node(
#                        'Add',
#                        inputs=[node1, bias_node],
#                        outputs=node.output('Out'))
#                else:
#                    node1 = graph.make_node('Add', inputs=[cast_node, bias_node])
#                    node2 = graph.make_node(
#                        'Mul',
#                        inputs=[node1, scale_node],
#                        outputs=[node.output('Out', 0)])
@op_mapper('scale')
class Scale():
    support_opset_version_range = (1, 12)

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
                'Constant',
                attrs={'dtype': dtypes.ONNX.FLOAT,
                       'value': [scale]})
            bias_node = graph.make_node(
                'Constant',
                attrs={'dtype': dtypes.ONNX.FLOAT,
                       'value': [bias]})
            cast_node = graph.make_node(
                'Cast', inputs=node.input('X'),
                attrs={'to': dtypes.ONNX.FLOAT})
            if node.attr('bias_after_scale'):
                node1 = graph.make_node('Mul', inputs=[cast_node, scale_node])
                node2 = graph.make_node(
                    'Add',
                    inputs=[node1, bias_node],
                    outputs=node.output('Out'))
            else:
                node1 = graph.make_node('Add', inputs=[cast_node, bias_node])
                node2 = graph.make_node(
                    'Mul',
                    inputs=[node1, scale_node],
                    outputs=[node.output('Out', 0)])


@op_mapper('softmax')
class Softmax():
    support_opset_version_range = (1, 12)

    @classmethod
    def opset_1(cls, graph, node, **kw):
        axis = node.attr('axis')
        shape = node.output_shape('Out', 0)
        if axis is None:
            axis = -1
        if axis < 0:
            axis += len(shape)
        if axis == len(shape) - 1:
            node = graph.make_node(
                'Softmax',
                inputs=node.input('X'),
                outputs=node.output('Out'),
                attrs={'axis': axis})
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


@op_mapper('softmax_with_cross_entropy')
class SoftmaxCrossEntropyLoss():
    support_opset_verison_range = (12, 12)

    @classmethod
    def opset_12(cls, graph, node, **kw):
        if node.attr('soft_label'):
            raise Exception(
                "SoftmaxCrossEntropyLoss in onnx not support soft label.")

        labels = node.input('Label', 0)
        scores = node.input('Logits', 0)

        outputs = [node.output('Loss', 0)]
        if 'Softmax' in node.outputs:
            outputs.append(node.output('Softmax', 0))

        shape = node.input_shape('Logits', 0)
        axis = node.attr('axis')
        if axis < 0:
            axis += len(shape)
        if axis == len(shape) - 1:
            graph.make_node(
                'SoftmaxCrossEntropyLoss',
                inputs=[scores, labels],
                outputs=outputs,
                ignore_index=node.attr('ignore_index'),
                reduction='mean')
        else:
            perm = [i for i in range(len(shape))]
            perm[-1] = axis
            perm[axis] = len(shape) - 1
            transpose_node = graph.make_node(
                'Transpose', inputs=node.input('X'), attrs={'perm': perm})
            node = graph.make_node(
                'SoftmaxCrossEntropyLoss',
                inputs=[scores, labels],
                outputs=outputs,
                ignore_index=node.attr('ignore_index'),
                reduction='mean')
            transpose_node1 = graph.make_node(
                'Transpose',
                inputs=[softmax_node],
                outputs=node.output('Out'),
                attrs={'perm': perm})
