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
import math
import collections
from paddle2onnx.constant import dtypes
from paddle2onnx.op_mapper import OpMapper as op_mapper
from paddle2onnx.op_mapper import mapper_helper
from paddle2onnx import utils
import paddle


@op_mapper(['conv2d', 'depthwise_conv2d', 'conv3d'])
class Conv():
    support_opset_version_range = (1, 12)

    @classmethod
    def opset_1(cls, graph, node, **kw):
        kernel_shape = node.input_shape('Filter', 0)
        dilations = node.attr('dilations')
        kernel_shape = kernel_shape[2:]
        strides = node.attr('strides')
        group = node.attr('groups')
        pads = node.attr('paddings')
        assert node.attrs['data_format'] == 'NCHW' or node.attrs['data_format'] == 'NCDHW' or node.attrs[
            'data_format'] == "AnyLayout", \
            "The conv data format should be 'NCHW' or 'NCDHW', but received data format " \
            "is %s." % node.attrs['data_format']
        # onnx padding is [x1_begin, x2_begin...x1_end, x2_end, ...]
        if len(pads) == 2 or len(pads) == 3:
            pads = pads + pads
        elif len(pads) == 4:
            pads = [pads[i] for i in [0, 2, 1, 3]]
        elif len(pads) == 6:
            pads = [pads[i] for i in [0, 2, 4, 1, 3, 5]]
        attrs = {
            'dilations': dilations,
            'kernel_shape': kernel_shape,
            'strides': strides,
            'group': group
        }
        auto_pad = node.attr('padding_algorithm')
        if auto_pad == 'SAME':
            attrs['auto_pad'] = 'SAME_UPPER'
        elif auto_pad == 'VALID':
            attrs['auto_pad'] = 'VALID'
        else:
            attrs['pads'] = pads
        graph.make_node(
            'Conv',
            inputs=node.input('Input') + node.input('Filter'),
            outputs=node.output('Output'),
            attrs=attrs)


@op_mapper(
    ['conv2d_transpose', 'depthwise_conv2d_transpose', 'conv3d_transpose'])
class ConvTranspose():
    support_opset_version_range = (1, 12)

    @classmethod
    def opset_1(cls, graph, node, **kw):
        output_padding = node.attr('output_padding')
        kernel_shape = node.input_shape('Filter', 0)
        dilations = node.attr('dilations')
        kernel_shape = kernel_shape[2:]
        strides = node.attr('strides')
        group = node.attr('groups')
        pads = node.attr('paddings')
        assert node.attrs['data_format'] == 'NCHW' or node.attrs['data_format'] == 'NCDHW', \
            "The conv data format should be 'NCHW' or 'NCDHW', but received data format " \
            "is %s." % node.attrs['data_format']

        if len(pads) == 2 or len(pads) == 3:
            pads = pads + pads
        elif len(pads) == 4:
            pads = [pads[i] for i in [0, 2, 1, 3]]
        elif len(pads) == 6:
            pads = [pads[i] for i in [0, 2, 4, 1, 3, 5]]

        attrs = {
            'dilations': dilations,
            'kernel_shape': kernel_shape,
            'strides': strides,
            'group': group
        }
        auto_pad = node.attr('padding_algorithm')
        if auto_pad == 'SAME':
            attrs['auto_pad'] = 'SAME_UPPER'
        elif auto_pad == 'VALID':
            attrs['auto_pad'] = 'VALID'
        else:
            attrs['pads'] = pads
        if output_padding and len(output_padding) > 0:
            attrs['output_padding'] = output_padding
        graph.make_node(
            'ConvTranspose',
            inputs=node.input('Input') + node.input('Filter'),
            outputs=node.output('Output'),
            attrs=attrs)


@op_mapper(
    ['pool2d', 'pool3d', 'max_pool2d_with_index', 'max_pool3d_with_index'])
class Pool():
    support_opset_version_range = (1, 15)
    pool_type = {
        'max': ('MaxPool', 'GlobalMaxPool'),
        'avg': ('AveragePool', 'GlobalAveragePool')
    }

    @classmethod
    def is_same_span(cls, in_size, out_size):
        spans = []
        for i in range(out_size):
            start = math.floor(i * (in_size / out_size))
            end = math.ceil((i + 1) * (in_size / out_size))
            spans.append(end - start)
        if len(set(spans)) == 1:
            return True
        return False

    @classmethod
    def opset_1(cls, graph, node, **kw):
        if node.type not in ["max_pool2d_with_index", "max_pool3d_with_index"]:
            assert node.attrs['data_format'] == 'NCHW' or node.attrs['data_format'] == 'NCDHW' or node.attrs[
                'data_format'] == "AnyLayout", \
                "The conv data format should be 'NCHW' or 'NCDHW', but received data format " \
                "is %s." % node.attrs['data_format']
        k_size = node.attr('ksize')
        if node.attr('global_pooling') or (
                node.attr('adaptive') and
            (k_size == [1, 1] or k_size == [1, 1, 1])):
            onnx_node = graph.make_node(
                cls.pool_type[node.attr('pooling_type')][1],
                inputs=node.input('X'),
                outputs=node.output('Out'))
        elif node.attr('adaptive'):
            cls.adaptive_pool(graph, node)
        else:
            cls.non_adaptive_pool(graph, node)

    @classmethod
    def non_adaptive_pool(cls, graph, node):
        input_shape = node.input_shape('X', 0)
        pads = node.attr('paddings')
        strides = node.attr('strides')
        k_size = node.attr('ksize')

        if len(pads) == 2 or len(pads) == 3:
            pads = pads + pads
        elif len(pads) == 4:
            pads = [pads[i] for i in [0, 2, 1, 3]]
        elif len(pads) == 6:
            pads = [pads[i] for i in [0, 2, 4, 1, 3, 5]]

        for i in range(len(input_shape) - 2):
            if input_shape[i + 2] > 0 and input_shape[i + 2] + pads[i] < k_size[
                    i]:
                k_size[i] = input_shape[i + 2] + pads[i]

        input_x = node.input('X')
        if max(k_size) <= max(pads):
            n = (int)(len(pads) / 2)
            onnx_paddings = [0, 0] + pads[:n] + [0, 0] + pads[n:]
            attrs_pad = {'mode': 'constant', }
            if graph.opset_version >= 11:
                pads_node = graph.make_node(
                    'Constant',
                    attrs={'dtype': dtypes.ONNX.INT64,
                           'value': onnx_paddings})
                value_node = graph.make_node(
                    'Constant',
                    attrs={'dtype': dtypes.ONNX.FLOAT,
                           'value': 0.0})
                input_x = input_x + [pads_node, value_node]
            else:
                attrs_pad['pads'] = onnx_paddings
                attrs_pad['value'] = 0.0
            input_x = graph.make_node('Pad', inputs=input_x, attrs=attrs_pad)
            pads = [0, 0] * n

        attrs = {
            'kernel_shape': k_size,
            'strides': strides,
        }
        auto_pad = node.attr('padding_algorithm')
        if auto_pad == 'SAME':
            attrs['auto_pad'] = 'SAME_UPPER'
        elif auto_pad == 'VALID':
            attrs['auto_pad'] = 'VALID'
        else:
            attrs['pads'] = pads
        if node.attr('ceil_mode') and graph.opset_version < 10:
            raise Exception(
                "Cannot convert pool with ceil_model == True to ONNX Opset version < 10"
            )
        elif node.attr('ceil_mode') is not None and graph.opset_version >= 10:
            attrs['ceil_mode'] = node.attr('ceil_mode')

        if node.attr('pooling_type') == 'avg':
            attrs['count_include_pad'] = not node.attr('exclusive')

        if node.type in ["max_pool2d_with_index", "max_pool3d_with_index"]:
            cls.get_max_pool2d_with_index(graph, node,
                                          len(k_size), input_x, attrs)
        else:
            onnx_node = graph.make_node(
                cls.pool_type[node.attr('pooling_type')][0],
                inputs=input_x,
                outputs=node.output('Out'),
                attrs=attrs)

    @classmethod
    def adaptive_pool(cls, graph, node):
        # if pool is adaptive, check if input shape of pool is fixed.
        mapper_helper.is_static_shape(node.input_shape('X', 0))
        input_shape = np.array(node.input_shape('X', 0)[2:])
        output_shape = np.array(node.output_shape('Out', 0)[2:])
        stride = input_shape // output_shape
        kernel = input_shape - (output_shape - 1) * stride

        for i in range(len(input_shape)):
            # check if kernel_size is fixed.
            if not cls.is_same_span(input_shape[i], output_shape[i]):
                raise Exception(
                    "Cannot convert adaptive pool with input_size: {}, output_size: {}"
                    .format(
                        node.input_shape('X', 0), node.output_shape('Out', 0)))

        attrs = {
            'kernel_shape': kernel,
            'strides': stride,
        }

        if node.attr('ceil_mode') and graph.opset_version < 10:
            raise Exception(
                "Cannot convert pool with ceil_model == True to ONNX Opset version < 10."
            )
        elif node.attr('ceil_mode') is not None and graph.opset_version > 10:
            attrs['ceil_mode'] = node.attr('ceil_mode')
        auto_pad = node.attr('padding_algorithm')
        if auto_pad == 'SAME':
            attrs['auto_pad'] = 'SAME_UPPER'
        elif auto_pad == 'VALID':
            attrs['auto_pad'] = 'VALID'
        if node.attr('pooling_type') == 'avg':
            attrs['count_include_pad'] = not node.attr('exclusive')
        onnx_node = graph.make_node(
            cls.pool_type[node.attr('pooling_type')][0],
            inputs=node.input('X'),
            outputs=node.output('Out'),
            attrs=attrs)

    @classmethod
    def get_max_pool2d_with_index(cls, graph, node, ndims, input, attrs):
        if graph.opset_version < 9:
            raise Exception(
                "Pool in onnx(opset<9) not support 'return index', Try converting with opset_version >=9"
            )
        mask_indices = graph.generate_node_name("pool")
        flattened_indices = graph.generate_node_name("pool")
        flattened_out = graph.generate_node_name("pool")
        graph.make_node(
            'MaxPool',
            inputs=input,
            outputs=[node.output('Out', 0), mask_indices],
            attrs=attrs)

        graph.make_node(
            'MaxPool',
            inputs=input,
            outputs=[flattened_out, flattened_indices],
            kernel_shape=[1] * ndims,
            strides=[1] * ndims)

        axes = [2 + i for i in range(ndims)]
        s = mapper_helper.slice_helper(
            graph,
            flattened_indices,
            axes=axes,
            starts=[0] * ndims,
            ends=[1] * ndims)
        onnx_node = graph.make_node("Sub", inputs=[mask_indices, s])
        graph.make_node(
            'Cast',
            inputs=[onnx_node],
            outputs=node.output('Mask'),
            attrs={
                'to': dtypes.DTYPE_PADDLE_ONNX_MAP[node.output_dtype('Mask', 0)]
            })


@op_mapper(['unpool'])
class UnPool():
    support_opset_version_range = (1, 15)
    pool_type = {'max': ('MaxUnpool', 'GlobalMaxPool'), }

    @classmethod
    def opset_1(cls, graph, node, **kw):
        input_shape = node.input_shape('X', 0)
        pads = node.attr('paddings')
        strides = node.attr('strides')
        k_size = node.attr('ksize')
        output_size = node.attr('output_size')

        if len(pads) == 2 or len(pads) == 3:
            pads = pads + pads
        elif len(pads) == 4:
            pads = [pads[i] for i in [0, 2, 1, 3]]
        elif len(pads) == 6:
            pads = [pads[i] for i in [0, 2, 4, 1, 3, 5]]

        for i in range(len(input_shape) - 2):
            if input_shape[i + 2] > 0 and input_shape[i + 2] + pads[i] < k_size[
                    i]:
                k_size[i] = input_shape[i + 2] + pads[i]

        input_x = node.input('X')
        input_idx = node.input('Indices')
        if max(k_size) <= max(pads):
            n = (int)(len(pads) / 2)
            onnx_paddings = [0, 0] + pads[:n] + [0, 0] + pads[n:]
            attrs_pad = {'mode': 'constant', }
            if graph.opset_version >= 11:
                pads_node = graph.make_node(
                    'Constant',
                    attrs={'dtype': dtypes.ONNX.INT64,
                           'value': onnx_paddings})
                value_node = graph.make_node(
                    'Constant',
                    attrs={'dtype': dtypes.ONNX.FLOAT,
                           'value': 0.0})
                input_x = input_x + [pads_node, value_node]
            else:
                attrs_pad['pads'] = onnx_paddings
                attrs_pad['value'] = 0.0
            input_x = graph.make_node('Pad', inputs=input_x, attrs=attrs_pad)
            pads = [0, 0] * n

        attrs = {
            'kernel_shape': k_size,
            'strides': strides,
        }
        auto_pad = node.attr('padding_algorithm')
        if auto_pad == 'SAME':
            attrs['auto_pad'] = 'SAME_UPPER'
        elif auto_pad == 'VALID':
            attrs['auto_pad'] = 'VALID'
        else:
            attrs['pads'] = pads
        if node.attr('ceil_mode') and graph.opset_version < 10:
            raise Exception(
                "Cannot convert pool with ceil_model == True to ONNX Opset version < 10"
            )
        elif node.attr('ceil_mode') is not None and graph.opset_version >= 10:
            attrs['ceil_mode'] = node.attr('ceil_mode')

        if node.attr('pooling_type') == 'avg':
            attrs['count_include_pad'] = not node.attr('exclusive')

        op = cls.pool_type[node.attr('unpooling_type')][0]
        idx_node = graph.make_node(
            'Cast', inputs=input_idx, attrs={'to': dtypes.ONNX.INT64})

        onnx_node = graph.make_node(
            op,
            inputs=[input_x[0], idx_node],
            outputs=node.output('Out'),
            attrs=attrs)


@op_mapper('elu')
class ELU():
    support_opset_version_range = (7, 15)

    @classmethod
    def opset_1(cls, graph, node, **kw):
        node = graph.make_node(
            'Elu',
            inputs=node.input('X'),
            outputs=node.output('Out'),
            alpha=node.attr('alpha'))


@op_mapper('softsign')
class SoftSign():
    support_opset_version_range = (7, 15)

    @classmethod
    def opset_1(cls, graph, node, **kw):
        graph.make_node(
            'Softsign', inputs=node.input('X'), outputs=node.output('Out'))


@op_mapper('hard_shrink')
class Hardshrink():
    support_opset_version_range = (9, 15)

    @classmethod
    def opset_9(cls, graph, node, **kw):
        node = graph.make_node(
            'Shrink',
            inputs=node.input('X'),
            outputs=node.output('Out'),
            lambd=node.attr('threshold'))


@op_mapper('logsigmoid')
class LogSigmoid():
    support_opset_version_range = (1, 12)

    @classmethod
    def opset_1(cls, graph, node, **kw):
        sigmoid_node = graph.make_node('Sigmoid', inputs=node.input('X'))
        graph.make_node('Log', inputs=sigmoid_node, outputs=node.output('Out'))


@op_mapper('norm')
class Norm():
    support_opset_version_range = (1, 12)

    @classmethod
    def opset_1(cls, graph, node, **kw):
        node = graph.make_node(
            'LpNormalization',
            inputs=node.input('X'),
            outputs=node.output('Out'),
            axis=node.attr('axis'))


@op_mapper('softshrink')
class SoftShrink():
    support_opset_version_range = (9, 15)

    @classmethod
    def opset_9(cls, graph, node, **kw):
        graph.make_node(
            'Shrink',
            inputs=node.input('X'),
            bias=node.attr('lambda'),
            lambd=node.attr('lambda'),
            outputs=node.output('Out'))


@op_mapper('tanh_shrink')
class TanhShrink():
    support_opset_version_range = (7, 15)

    @classmethod
    def opset_7(cls, graph, node, **kw):
        tanh_node = graph.make_node(
            'Tanh',
            inputs=node.input('X', 0), )
        graph.make_node(
            'Sub',
            inputs=[node.input('X', 0), tanh_node],
            outputs=node.output('Out'))


@op_mapper('log_softmax')
class LogSoftmax():
    support_opset_version_range = (7, 15)

    @classmethod
    def opset_7(cls, graph, node, **kw):
        axis = node.attr('axis')
        shape = node.output_shape('Out', 0)
        if axis is None:
            axis = -1
        if axis < 0:
            axis += len(shape)
        if axis == len(shape) - 1:
            node = graph.make_node(
                'LogSoftmax',
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
                'LogSoftmax', inputs=[transpose_node], axis=-1)
            transpose_node1 = graph.make_node(
                'Transpose',
                inputs=[softmax_node],
                outputs=node.output('Out'),
                attrs={'perm': perm})

    @classmethod
    def opset_13(cls, graph, node, **kw):
        graph.make_node(
            'LogSoftmax',
            inputs=node.input('X'),
            axis=node.attr('axis'),
            outputs=node.output('Out'))


@op_mapper('layer_norm')
class LayerNorm():
    support_opset_version_range = (7, 15)

    @classmethod
    def opset_7(cls, graph, node, **kw):
        ipt = node.input('X', 0)
        ipt_dims = len(node.input_shape('X', 0))
        normalized_shape = node.attr('begin_norm_axis')
        axes = None
        if isinstance(normalized_shape, collections.Iterable):
            axes = [-i for i in range(len(normalized_shape), 0, -1)]
        else:
            axes = [i for i in range(normalized_shape, ipt_dims)]
        dtype = node.block.vars[node.input('X', 0)].dtype
        dtype = dtypes.DTYPE_PADDLE_ONNX_MAP[dtype]
        epsilon = graph.make_node(
            'Constant', dtype=dtype, value=node.attr('epsilon'))
        two = graph.make_node('Constant', dtype=dtype, value=2.0)
        mean = graph.make_node("ReduceMean", inputs=[ipt], axes=axes)
        numerator = graph.make_node("Sub", inputs=[ipt, mean])
        pow_num = graph.make_node("Pow", inputs=[numerator, two])
        variance = graph.make_node("ReduceMean", inputs=[pow_num], axes=axes)
        add_eps = graph.make_node("Add", inputs=[variance, epsilon])
        denominator = graph.make_node("Sqrt", inputs=[add_eps])

        if 'Bias' in node.inputs and 'Scale' in node.inputs and len(
                node.input('Scale')) > 0 and len(node.input('Bias')) > 0:
            ipt_shape = graph.make_node("Shape", inputs=[ipt])
            weight_shape = mapper_helper.slice_helper(
                graph, ipt_shape, [0], [ipt_dims - len(axes)], [ipt_dims])
            scale = graph.make_node(
                "Reshape", inputs=[node.input('Scale', 0), weight_shape])
            bias = graph.make_node(
                "Reshape", inputs=[node.input('Bias', 0), weight_shape])
            layer_norm = graph.make_node("Div", inputs=[numerator, denominator])
            layer_norm = graph.make_node("Mul", inputs=[layer_norm, scale])
            graph.make_node(
                "Add", inputs=[layer_norm, bias], outputs=node.output('Y'))
        elif 'Bias' in node.inputs and len(node.input('Bias')) > 0:
            ipt_shape = graph.make_node("Shape", inputs=[ipt])
            weight_shape = mapper_helper.slice_helper(
                graph, ipt_shape, [0], [ipt_dims - len(axes)], [ipt_dims])
            bias = graph.make_node(
                "Reshape", inputs=[node.input('Bias', 0), weight_shape])
            layer_norm = graph.make_node("Div", inputs=[numerator, denominator])
            graph.make_node(
                "Add", inputs=[layer_norm, bias], outputs=node.output('Y'))
        elif 'Scale' in node.inputs and len(node.input('Scale')) > 0:
            ipt_shape = graph.make_node("Shape", inputs=[ipt])
            weight_shape = mapper_helper.slice_helper(
                graph, ipt_shape, [0], [ipt_dims - len(axes)], [ipt_dims])
            scale = graph.make_node(
                "Reshape", inputs=[node.input('Scale', 0), weight_shape])
            layer_norm = graph.make_node("Div", inputs=[numerator, denominator])
            graph.make_node(
                "Mul", inputs=[layer_norm, scale], outputs=node.output('Y'))
        else:
            layer_norm = graph.make_node(
                "Div",
                inputs=[numerator, denominator],
                outputs=node.output('Y'))


@op_mapper('batch_norm')
class BatchNorm():
    support_opset_version_range = (7, 15)

    @classmethod
    def make_attrs_and_inputs(cls, graph, node, **kw):
        onnx_attr = {
            'epsilon': node.attr('epsilon'),
            'momentum': node.attr('momentum')
        }
        inputs = node.input('X') + node.input('Scale') + node.input(
            'Bias') + node.input('Mean') + node.input('Variance')
        return onnx_attr, inputs

    @classmethod
    def opset_9(cls, graph, node, **kw):
        onnx_attr, inputs = cls.make_attrs_and_inputs(graph, node, **kw)
        onnx_node = graph.make_node(
            'BatchNormalization',
            inputs=inputs,
            outputs=node.output('Y'),
            **onnx_attr)

    @classmethod
    def opset_7(cls, graph, node, **kw):
        onnx_attr, inputs = cls.make_attrs_and_inputs(graph, node, **kw)
        onnx_attr['spatial'] = 1
        onnx_node = graph.make_node(
            'BatchNormalization',
            inputs=inputs,
            outputs=node.output('Y'),
            **onnx_attr)


@op_mapper('group_norm')
class GroupNorm():
    support_opset_version_range = (6, 15)

    @classmethod
    def opset_13(cls, graph, node, **kw):
        num_groups = node.attr('groups')
        epsilon = node.attr('epsilon')
        ipt = node.input('X')[0]

        ipt_shape = node.input_shape('X', 0)
        assert len(
            ipt_shape) == 4, "Only support 4D-Tensor as input for GroupNorm"

        shape = graph.make_node(
            'Constant', dtype=dtypes.ONNX.INT64, value=[0, num_groups, -1])
        reshape_input = graph.make_node('Reshape', inputs=[ipt, shape])
        scale_ = graph.make_node(
            'Constant', dtype=dtypes.ONNX.FLOAT, value=[1.0] * num_groups)
        bias_ = graph.make_node(
            'Constant', dtype=dtypes.ONNX.FLOAT, value=[0.0] * num_groups)
        reshaped_output = graph.make_node(
            'InstanceNormalization',
            inputs=[reshape_input, scale_, bias_],
            epsilon=epsilon)
        origin_shape = graph.make_node('Shape', inputs=[ipt])

        if len(node.input('Scale')) > 0 and len(node.input('Bias')) > 0:
            output = graph.make_node(
                'Reshape', inputs=[reshaped_output, origin_shape])
            axes = graph.make_node(
                'Constant', dtype=dtypes.ONNX.INT64, value=[1, 2])
            scale = node.input('Scale')[0]
            bias = node.input('Bias')[0]
            unsqueezed_scale = graph.make_node(
                'Unsqueeze', inputs=[scale, axes])
            unsqueezed_bias = graph.make_node('Unsqueeze', inputs=[bias, axes])
            part0 = graph.make_node('Mul', inputs=[output, unsqueezed_scale])
            graph.make_node(
                'Add',
                inputs=[part0, unsqueezed_bias],
                outputs=node.output('Y'))
        else:
            output = graph.make_node(
                'Reshape',
                inputs=[reshaped_output, origin_shape],
                outputs=node.output('Y'))

    @classmethod
    def opset_6(cls, graph, node, **kw):
        num_groups = node.attr('groups')
        epsilon = node.attr('epsilon')
        ipt = node.input('X')[0]

        ipt_shape = node.input_shape('X', 0)
        assert len(
            ipt_shape) == 4, "Only support 4D-Tensor as input for GroupNorm"

        dtype = node.block.vars[node.input('X', 0)].dtype
        dtype = dtypes.DTYPE_PADDLE_ONNX_MAP[dtype]

        shape = graph.make_node(
            'Constant', dtype=dtypes.ONNX.INT64, value=[0, num_groups, -1])
        reshape_input = graph.make_node('Reshape', inputs=[ipt, shape])
        scale_ = graph.make_node(
            'Constant', dtype=dtype, value=[1.0] * num_groups)
        bias_ = graph.make_node(
            'Constant', dtype=dtype, value=[0.0] * num_groups)
        reshaped_output = graph.make_node(
            'InstanceNormalization',
            inputs=[reshape_input, scale_, bias_],
            epsilon=epsilon)
        origin_shape = graph.make_node('Shape', inputs=[ipt])

        if len(node.input('Scale')) > 0 and len(node.input('Bias')) > 0:
            output = graph.make_node(
                'Reshape', inputs=[reshaped_output, origin_shape])
            scale = node.input('Scale')[0]
            bias = node.input('Bias')[0]
            unsqueezed_scale = graph.make_node(
                'Unsqueeze', inputs=[scale], axes=[1, 2])
            unsqueezed_bias = graph.make_node(
                'Unsqueeze', inputs=[bias], axes=[1, 2])
            part0 = graph.make_node('Mul', inputs=[output, unsqueezed_scale])
            graph.make_node(
                'Add',
                inputs=[part0, unsqueezed_bias],
                outputs=node.output('Y'))
        else:
            output = graph.make_node(
                'Reshape',
                inputs=[reshaped_output, origin_shape],
                outputs=node.output('Y'))


@op_mapper('instance_norm')
class InstanceNorm():
    support_opset_version_range = (6, 15)

    @classmethod
    def opset_6(cls, graph, node, **kw):
        onnx_attr = {'epsilon': node.attr('epsilon'), }
        num_groups = node.block.vars[node.input('X')[0]].shape[1]

        dtype = node.block.vars[node.input('X', 0)].dtype
        dtype = dtypes.DTYPE_PADDLE_ONNX_MAP[dtype]

        if len(node.input('Scale')) == 0:
            scale_ = graph.make_node(
                'Constant', dtype=dtype, value=[1.0] * num_groups)
        else:
            scale_ = node.input('Scale')[0]
        if len(node.input('Bias')) == 0:
            bias_ = graph.make_node(
                'Constant', dtype=dtype, value=[0.0] * num_groups)
        else:
            bias_ = node.input('Bias')[0]

        inputs = node.input('X') + [scale_] + [bias_]
        onnx_node = graph.make_node(
            'InstanceNormalization',
            inputs=inputs,
            outputs=node.output('Y'),
            **onnx_attr)


@op_mapper('dropout')
class Dropout():
    support_opset_version_range = (7, 15)

    @classmethod
    def opset_7(cls, graph, node, **kw):
        dropout_mode = node.attr('dropout_implementation')
        dropout_prob = node.attr('dropout_prob')
        if dropout_mode == 'upscale_in_train':
            onnx_node = graph.make_node(
                'Identity', inputs=node.input('X'), outputs=node.output('Out'))
        elif dropout_mode == 'downgrade_in_infer':
            scale_node = graph.make_node(
                'Constant',
                attrs={'dtype': dtypes.ONNX.FLOAT,
                       'value': 1 - dropout_prob})
            graph.make_node(
                "Mul",
                inputs=[node.input('X')[0], scale_node],
                outputs=node.output('Out'))
        else:
            raise Exception("Unexpected situation happend")


@op_mapper('roi_align')
class RoiAlign():
    support_opset_version_range = (10, 12)

    @classmethod
    def opset_10(cls, graph, node, **kw):
        rois_shape = graph.make_node('Shape', inputs=[node.input('ROIs', 0)])
        starts = graph.make_node(
            'Constant', attrs={'dtype': dtypes.ONNX.INT64,
                               'value': [0]})
        ends = graph.make_node(
            'Constant', attrs={'dtype': dtypes.ONNX.INT64,
                               'value': [1]})
        num_rois = graph.make_node('Slice', inputs=[rois_shape, starts, ends])
        zero = graph.make_node(
            'Constant', dims=[1], dtype=dtypes.ONNX.INT64, value=[0])
        batch_indices = graph.make_node('Expand', inputs=[zero, num_rois])
        node = graph.make_node(
            'RoiAlign',
            inputs=[node.input('X', 0), node.input('ROIs', 0), batch_indices],
            outputs=node.output('Out'),
            mode='avg',
            output_height=node.attr('pooled_height'),
            output_width=node.attr('pooled_width'),
            sampling_ratio=node.attr('sampling_ratio'),
            spatial_scale=node.attr('spatial_scale'))


@op_mapper('rnn')
class RNN():
    support_opset_version_range = (1, 12)

    @classmethod
    def make_param_inputs(cls, graph, node, layer, hidden_size, num_layers):
        # weight assign order:
        # (F_whi F_whh B_whi B_whh）* layer_num  + (F_bias_hi F_bias_hh B_bias_hi  B_bias_hi)* layer_num
        def reform_weights(g, w, n, intervals):
            slices = [
                mapper_helper.slice_helper(
                    g, w, axes=[1], starts=[x * n], ends=[y * n])
                for x, y in intervals
            ]
            return g.make_node('Concat', slices, axis=1)

        def transform_weight_with_bias(g, weights, n, intervals):
            return [reform_weights(g, w, n, intervals) for w in weights]

        if node.attr('mode') == 'LSTM':
            reform_permutation = [(0, 1), (3, 4), (1, 3)]
        elif node.attr('mode') == 'GRU':
            reform_permutation = [(1, 2), (0, 1), (2, 3)]
        bidirect_len = 4 if node.attr('is_bidirec') else 2
        all_layer_param_len = len(node.input('WeightList'))
        weight_list = node.input('WeightList')[:all_layer_param_len // 2]
        bias_list = node.input('WeightList')[all_layer_param_len // 2:]
        single_layer_param_len = all_layer_param_len // num_layers

        unsqueeze_weights = []
        layer_weight_list = weight_list[layer * bidirect_len:layer *
                                        bidirect_len + bidirect_len]
        layer_bias_list = bias_list[layer * bidirect_len:layer * bidirect_len +
                                    bidirect_len]
        param_list = layer_weight_list + layer_bias_list
        param_list_len = len(param_list)
        for i in range(param_list_len):
            weight = graph.make_node(
                'Unsqueeze', inputs=[param_list[i]], axes=[0])
            unsqueeze_weights.append(weight)

        input_weights = unsqueeze_weights[0:param_list_len // 2:2]
        hidden_weights = unsqueeze_weights[1:param_list_len // 2:2]

        input_weight = graph.make_node('Concat', inputs=input_weights, axis=0)
        hidden_weight = graph.make_node('Concat', inputs=hidden_weights, axis=0)
        input_bias = unsqueeze_weights[param_list_len // 2:param_list_len:2]
        hidden_bias = unsqueeze_weights[param_list_len // 2 + 1:param_list_len:
                                        2]

        input_bias = graph.make_node('Concat', inputs=input_bias, axis=0)
        hidden_bias = graph.make_node('Concat', inputs=hidden_bias, axis=0)
        input_weight, hidden_weight, input_bias, hidden_bias = transform_weight_with_bias(
            graph, [input_weight, hidden_weight, input_bias, hidden_bias],
            hidden_size, reform_permutation)
        bias = graph.make_node(
            'Concat', inputs=[input_bias, hidden_bias], axis=1)
        return [input_weight, hidden_weight, bias, '']

    @classmethod
    def make_init_param_inputs(cls, graph, node, layer):
        if node.attr('mode') == 'LSTM':
            all_init_h, all_init_c = node.input('PreState')
            bidirect_len = 2 if node.attr('is_bidirec') else 1
            init_h = mapper_helper.slice_helper(
                graph, all_init_h, [0], [layer * bidirect_len],
                [layer * bidirect_len + bidirect_len])
            init_c = mapper_helper.slice_helper(
                graph, all_init_c, [0], [layer * bidirect_len],
                [layer * bidirect_len + bidirect_len])
            return [init_h, init_c]
        elif node.attr('mode') == 'GRU':
            all_init_h = node.input('PreState', 0)
            bidirect_len = 2 if node.attr('is_bidirec') else 1
            init_h = mapper_helper.slice_helper(
                graph, all_init_h, [0], [layer * bidirect_len],
                [layer * bidirect_len + bidirect_len])
            return [init_h]

    @classmethod
    def opset_9(cls, graph, node, **kw):
        mode = node.attr('mode')
        hidden_size = node.attr('hidden_size')
        num_layers = node.attr('num_layers')
        prev_output = node.input('Input', 0)
        if node.attr('mode') == 'LSTM':
            for layer in range(num_layers):
                param_inputs = cls.make_param_inputs(graph, node, layer,
                                                     hidden_size, num_layers)
                init_param_inputs = cls.make_init_param_inputs(graph, node,
                                                               layer)
                if layer + 1 < num_layers:
                    rnn_outputs = 3
                    output_y = None
                else:
                    rnn_outputs = [1] + node.output('State')
                    output_y = node.output('Out')
                prev_output, h_out, c_out = graph.make_node(
                    node.attr('mode'),
                    inputs=[prev_output] + param_inputs + init_param_inputs,
                    outputs=rnn_outputs,
                    direction='bidirectional'
                    if node.attr('is_bidirec') else 'forward',
                    hidden_size=node.attr('hidden_size'))
                prev_output = graph.make_node(
                    'Transpose', inputs=[prev_output], perm=[0, 2, 1, 3])

                prev_shape = graph.make_node(
                    'Constant', dtype=dtypes.ONNX.INT64, value=[0, 0, -1])
                prev_output = graph.make_node(
                    'Reshape',
                    inputs=[prev_output, prev_shape],
                    outputs=output_y)
        elif node.attr('mode') == 'GRU':
            for layer in range(num_layers):
                param_inputs = cls.make_param_inputs(graph, node, layer,
                                                     hidden_size, num_layers)
                init_param_inputs = cls.make_init_param_inputs(graph, node,
                                                               layer)
                if layer + 1 < num_layers:
                    rnn_outputs = 2
                    output_y = None
                else:
                    rnn_outputs = [1] + node.output('State')
                    output_y = node.output('Out')
                attrs = {
                    'direction': 'bidirectional'
                    if node.attr('is_bidirec') else 'forward',
                    'hidden_size': node.attr('hidden_size'),
                    'linear_before_reset': 1,
                }
                prev_output, h_out = graph.make_node(
                    node.attr('mode'),
                    inputs=[prev_output] + param_inputs + init_param_inputs,
                    outputs=rnn_outputs,
                    attrs=attrs)
                prev_output = graph.make_node(
                    'Transpose', inputs=[prev_output], perm=[0, 2, 1, 3])
                prev_shape = graph.make_node(
                    'Constant', dtype=dtypes.ONNX.INT64, value=[0, 0, -1])
                prev_output = graph.make_node(
                    'Reshape',
                    inputs=[prev_output, prev_shape],
                    outputs=output_y)


@op_mapper('thresholded_relu')
class ThresholdedRelu():
    support_opset_version_range = (10, 15)

    @classmethod
    def opset_10(cls, graph, node, **kw):
        x_dtype = node.input_dtype('X', 0)
        if x_dtype != paddle.float32:
            x = graph.make_node(
                'Cast', inputs=node.input('X'), to=dtypes.ONNX.FLOAT)
            threshholdedrelu_node = graph.make_node(
                'ThresholdedRelu', inputs=[x], alpha=node.attr('threshold'))
            graph.make_node(
                'Cast',
                inputs=[threshholdedrelu_node],
                outputs=node.output('Out'),
                to=dtypes.DTYPE_PADDLE_ONNX_MAP[x_dtype])
        else:
            graph.make_node(
                'ThresholdedRelu',
                inputs=node.input('X'),
                alpha=node.attr('threshold'),
                outputs=node.output('Out'))
