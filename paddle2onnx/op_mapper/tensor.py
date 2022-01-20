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
import copy
import paddle


@op_mapper('concat')
class Concat():
    support_opset_version_range = (4, 15)

    @classmethod
    def opset_4(cls, graph, node, **kw):
        inputs = node.input('X')

        input_dtypes = [node.input_dtype('X', i) for i in range(len(inputs))]
        inputs = mapper_helper.dtype_alignment(graph, inputs, input_dtypes)
        if len(node.input('AxisTensor')) > 0:
            axis_node = node.input('AxisTensor')[0]
            # When axis is tensor, only int32 and int64 are supported
            if axis_node not in graph.parameters:
                raise Exception(
                    "Currently does not support the axis parameter as input tensor!"
                )
            else:
                axis = graph.parameters[axis_node].attribute[0].t.int32_data
                if axis is None or len(axis) < 1:
                    axis = graph.parameters[axis_node].attribute[
                        0].t.int64_data[0]
        else:
            axis = node.attr('axis')
        if axis < 0:
            axis = axis + len(node.input_shape('X', 0))

        node = graph.make_node(
            'Concat', inputs=inputs, outputs=node.output('Out'), axis=axis)


@op_mapper('assign')
class Assign():
    support_opset_version_range = (1, 12)

    @classmethod
    def opset_1(cls, graph, node, **kw):
        inputs = node.input('X')
        graph.make_node('Identity', inputs=inputs, outputs=node.output('Out'))


@op_mapper('lod_reset')
class LodReset():
    support_opset_version_range = (1, )

    @classmethod
    def opset_1(cls, graph, node, **kw):
        graph.make_node(
            'Identity', inputs=node.input('X'), outputs=node.output('Out'))


@op_mapper('stack')
class Stack():
    support_opset_version_range = (4, 15)

    @classmethod
    def opset_4(cls, graph, node, **kw):
        inputs = node.input('X')
        input_dtypes = [node.input_dtype('X', i) for i in range(len(inputs))]
        inputs = mapper_helper.dtype_alignment(graph, inputs, input_dtypes)
        axis = node.attr('axis')

        unsqueezed_inputs = list()
        for ipt in inputs:
            unsqueezed_ipt = graph.make_node(
                'Unsqueeze', inputs=[ipt], axes=[axis])
            unsqueezed_inputs.append(unsqueezed_ipt)
        graph.make_node(
            'Concat',
            inputs=unsqueezed_inputs,
            outputs=node.output('Y'),
            axis=axis)

    @classmethod
    def opset_13(cls, graph, node, **kw):
        inputs = node.input('X')
        input_dtypes = [node.input_dtype('X', i) for i in range(len(inputs))]
        inputs = mapper_helper.dtype_alignment(graph, inputs, input_dtypes)
        axis = node.attr('axis')

        unsqueezed_inputs = list()
        for ipt in inputs:
            axes_node = graph.make_node(
                'Constant', attrs={'dtype': dtypes.ONNX.INT64,
                                   'value': axis})
            unsqueezed_ipt = graph.make_node(
                'Unsqueeze', inputs=[ipt, axes_node])
            unsqueezed_inputs.append(unsqueezed_ipt)
        graph.make_node(
            'Concat',
            inputs=unsqueezed_inputs,
            outputs=node.output('Y'),
            axis=axis)


@op_mapper('unstack')
class Unstack():
    support_opset_version_range = (1, 12)

    @classmethod
    def opset_1(cls, graph, node, **kw):
        print(node)
        graph.make_node(
            'Split',
            inputs=node.input('X'),
            outputs=node.output('Y'),
            axis=node.attr('axis'))


@op_mapper('expand_as_v2')
class ExpandAsV2():
    support_opset_version_range = (8, 12)

    @classmethod
    def opset_8(cls, graph, node, **kw):
        target_shape = node.attr('target_shape')
        if node.input('target_tensor', 0) is not None:
            target_shape = graph.make_node(
                'Shape', inputs=[node.input('target_tensor', 0)])
        elif target_shape is not None:
            target_shape = graph.make_node(
                'Constant',
                attrs={'dtype': dtypes.ONNX.INT64,
                       'value': target_shape})
        else:
            raise Exception(
                "Not find attribute: 'target_shape' or tensor 'target_tensor'")
        node = graph.make_node(
            'Expand',
            inputs=[node.input('X', 0), target_shape],
            outputs=node.output('Out'))


@op_mapper('expand_v2')
class ExpandV2():
    support_opset_version_range = (8, 12)

    @classmethod
    def opset_8(cls, graph, node, **kw):
        if node.input('expand_shapes_tensor') is not None and len(
                node.input('expand_shapes_tensor')) > 0:
            shape = graph.make_node(
                'Concat', inputs=node.input('expand_shapes_tensor'), axis=-1)
            shape = graph.make_node(
                'Cast', inputs=[shape], to=dtypes.ONNX.INT64)
        else:
            if len(node.input('Shape')) > 0:
                shape = mapper_helper.cast(graph,
                                           node.input('Shape', 0),
                                           node.input_dtype('Shape', 0),
                                           'int64')
            elif len(node.attr('shape')) > 0:
                shape = node.attr('shape')
                for idx in range(len(shape)):
                    if shape[idx] == -1:
                        shape[idx] = 1
                shape = graph.make_node(
                    'Constant', dtype=dtypes.ONNX.INT64, value=shape)
        node = graph.make_node(
            'Expand',
            inputs=[node.input('X', 0), shape],
            outputs=node.output('Out'))


@op_mapper('shape')
class Shape():
    support_opset_version_range = (6, 15)

    @classmethod
    def opset_6(cls, graph, node, **kw):
        shape_node = graph.make_node('Shape', inputs=node.input('Input'))
        graph.make_node(
            'Cast',
            inputs=[shape_node],
            outputs=node.output('Out'),
            to=dtypes.ONNX.INT32)


@op_mapper('size')
class Numel():
    supports_opset_version_range = (1, 15)

    @classmethod
    def opset_1(cls, graph, node, **kw):
        size_node = graph.make_node('Size', inputs=node.input('Input'))
        graph.make_node(
            'Unsqueeze', inputs=size_node, axes=[0], outputs=node.output('Out'))

    @classmethod
    def opset_13(cls, graph, node, **kw):
        size_node = graph.make_node('Size', inputs=node.input('Input'))
        axes_node = graph.make_node(
            'Constant', attrs={'dtype': dtypes.ONNX.INT64,
                               'value': [0]})
        graph.make_node(
            'Unsqueeze',
            inputs=[size_node, axes_node],
            outputs=node.output('Out'))


@op_mapper('split')
class Split():
    support_opset_version_range = (1, 15)

    @classmethod
    def opset_1(cls, graph, node, **kw):
        sections = node.attr('sections')
        axis = cls.get_axis(graph, node)
        if len(sections) > 0:
            input_shape = node.block.vars[node.input('X')[0]].shape
            input_index = [i for i, val in enumerate(input_shape) if val == -1]
            section_index = [i for i, val in enumerate(sections) if val == -1]
            if len(input_index) == 0 and len(section_index) == 1:
                sections[section_index[0]] = input_shape[axis] - sum(
                    sections) - 1
            graph.make_node(
                'Split',
                inputs=node.input('X'),
                outputs=node.output('Out'),
                axis=axis,
                split=sections)
        else:
            graph.make_node(
                'Split',
                inputs=node.input('X'),
                outputs=node.output('Out'),
                axis=axis)

    @classmethod
    def opset_13(cls, graph, node, **kw):
        sections = node.attr('sections')
        axis = cls.get_axis(graph, node)
        if len(sections) > 0:
            input_shape = node.block.vars[node.input('X')[0]].shape
            input_index = [i for i, val in enumerate(input_shape) if val == -1]
            section_index = [i for i, val in enumerate(sections) if val == -1]
            if len(input_index) == 0 and len(section_index) == 1:
                sections[section_index[0]] = input_shape[axis] - sum(
                    sections) - 1
            split_node = graph.make_node(
                'Constant',
                attrs={'dtype': dtypes.ONNX.INT64,
                       'value': sections})
            graph.make_node(
                'Split',
                inputs=[node.input('X')[0], split_node],
                outputs=node.output('Out'),
                axis=axis)
        else:
            graph.make_node(
                'Split',
                inputs=node.input('X'),
                outputs=node.output('Out'),
                axis=axis)

    @classmethod
    def get_axis(cls, graph, node):
        if len(node.input('AxisTensor')) > 0:
            axis_node = node.input('AxisTensor')[0]
            # When axis is tensor, only int32 and int64 are supported
            if axis_node not in graph.parameters:
                raise Exception(
                    "Currently does not support the axis parameter as input tensor!"
                )
            else:
                axis = graph.parameters[axis_node].attribute[0].t.int32_data
                if axis is None or len(axis) < 1:
                    axis = graph.parameters[axis_node].attribute[
                        0].t.int64_data[0]
        else:
            axis = node.attr('axis')
        return axis


@op_mapper(['slice', 'strided_slice'])
class Slice():
    support_opset_version_range = (1, 12)

    @classmethod
    def decrease_axis(cls, node):
        # tensor[i,:] will decrease rank of origin input, example:
        # paddle.slice() will not decrease rank of origin input
        # if input shape is [2, 3], input[0, :] will generate output with shape [3], not [1, 3].
        # paddle.slice(input, 0, 1, 0) will  generate output with shape [1, 3], not [3]. 

        decrease_axis = node.attr('decrease_axis')
        if len(decrease_axis) == 0:
            return None
        if node.output_shape('Out', 0) == [0]:
            return decrease_axis
        if len(node.input_shape('Input', 0)) > len(node.output_shape('Out', 0)):
            return decrease_axis
        return None

    @classmethod
    def opset_1(cls, graph, node, **kw):
        axes = node.attr('axes')
        starts = node.attr('starts')
        ends = node.attr('ends')
        steps = node.attr('strides', [1] * len(ends))

        input_shape = node.input_shape('Input', 0)
        for i, e in enumerate(ends):
            axis = axes[i]
            if e > input_shape[axis] and input_shape[axis] > 0:
                ends[i] = input_shape[axis]

        if steps != [1] * len(ends):
            raise Exception(
                "Slice in onnx(opset<10) not support attribute 'step', Try converting with opset_version >=10"
            )
        decrease_axis = cls.decrease_axis(node)
        if decrease_axis is None:
            graph.make_node(
                "Slice",
                inputs=[node.input('Input')[0]],
                outputs=node.output('Out'),
                axes=axes,
                starts=starts,
                ends=ends)
        else:
            sliced = graph.make_node(
                "Slice",
                inputs=[node.input('Input')[0]],
                axes=axes,
                starts=starts,
                ends=ends)
            graph.make_node(
                'Squeeze',
                inputs=[sliced],
                outputs=node.output('Out'),
                axes=decrease_axis)

    @classmethod
    def opset_10(cls, graph, node, **kw):
        axes = node.attr('axes')
        starts = node.attr('starts')
        ends = node.attr('ends')
        steps = node.attr('strides', [1] * len(ends))

        input_shape = node.input_shape('Input', 0)
        for i, e in enumerate(ends):
            axis = axes[i]
            if e > input_shape[axis] and input_shape[axis] > 0:
                ends[i] = input_shape[axis]

        for i, s in enumerate(starts):
            axis = axes[i]
            if s < 0 and input_shape[axis] > 0:
                starts[i] = input_shape[axis] + s

        axes_node = graph.make_node(
            'Constant', attrs={'dtype': dtypes.ONNX.INT64,
                               'value': axes})
        starts_node = graph.make_node(
            'Constant', attrs={'dtype': dtypes.ONNX.INT64,
                               'value': starts})
        ends_node = graph.make_node(
            'Constant', attrs={'dtype': dtypes.ONNX.INT64,
                               'value': ends})
        steps_node = graph.make_node(
            'Constant', attrs={'dtype': dtypes.ONNX.INT64,
                               'value': steps})

        decrease_axis = cls.decrease_axis(node)
        if decrease_axis is None:
            sliced = graph.make_node(
                "Slice",
                inputs=[
                    node.input('Input')[0], starts_node, ends_node, axes_node,
                    steps_node
                ],
                outputs=node.output('Out'))
        else:
            sliced = graph.make_node(
                "Slice",
                inputs=[
                    node.input('Input')[0], starts_node, ends_node, axes_node,
                    steps_node
                ])
            graph.make_node(
                'Squeeze',
                inputs=[sliced],
                outputs=node.output('Out'),
                axes=decrease_axis)


@op_mapper(['sequence_expand'])
class SequenceExpand():
    support_opset_version_range = ()

    @classmethod
    def opset_1(cls, graph, node, **kw):
        graph.make_node(
            'Identity', inputs=node.input('X'), outputs=node.output('Out'))


@op_mapper(['expand', 'tile'])
class Expand():
    support_opset_version_range = (11, 12)

    @classmethod
    def opset_11(cls, graph, node, **kw):
        expand_times = node.attr('expand_times')
        if expand_times is None:
            expand_times = node.attr('repeat_times')

        if 'repeat_times_tensor' in node.inputs and len(
                node.input('repeat_times_tensor')) > 0:
            if len(node.input('repeat_times_tensor')) > 1:
                repeat_times = node.input('repeat_times_tensor')
                repeat_times_dtypes = [
                    node.input_dtype('repeat_times_tensor', i)
                    for i in range(len(repeat_times))
                ]
                repeat_times = mapper_helper.dtype_alignment(
                    graph, repeat_times, repeat_times_dtypes)

                # When OpSet>=11, Concat could use negative axis
                repeat_times_tensor = graph.make_node(
                    'Concat', inputs=repeat_times, axis=-1)
                graph.make_node(
                    "Tile",
                    inputs=[node.input('X', 0), repeat_times_tensor],
                    outputs=node.output('Out'))
            else:
                graph.make_node(
                    "Tile",
                    inputs=[
                        node.input('X', 0), node.input('repeat_times_tensor', 0)
                    ],
                    outputs=node.output('Out'))
        elif 'RepeatTimes' in node.inputs and len(node.input(
                'RepeatTimes')) == 1:
            repeat_times = mapper_helper.cast(graph,
                                              node.input('RepeatTimes', 0),
                                              node.input_dtype('RepeatTimes',
                                                               0), 'int64')
            graph.make_node(
                "Tile",
                inputs=[node.input('X', 0), repeat_times],
                outputs=node.output('Out'))
        elif expand_times is None:
            raise Exception("Not find attribute: 'repeat_times'.")
        elif -1 not in expand_times:
            expand_times_node = graph.make_node(
                'Constant',
                attrs={'dtype': dtypes.ONNX.INT64,
                       'value': expand_times})
            graph.make_node(
                "Tile",
                inputs=[node.input('X', 0), expand_times_node],
                outputs=node.output('Out'))
        else:
            raise Exception("illegal Tensor: 'repeat_times'.")


@op_mapper('range')
class Range():
    support_opset_version_range = (11, 12)

    @classmethod
    def opset_11(cls, graph, node, **kw):
        start = node.input('Start', 0)
        end = node.input('End', 0)
        step = node.input('Step', 0)
        start_t = mapper_helper.squeeze_helper(graph, start, [0])
        end_t = mapper_helper.squeeze_helper(graph, end, [0])
        step_t = mapper_helper.squeeze_helper(graph, step, [0])
        graph.make_node(
            "Range",
            inputs=[start_t, end_t, step_t],
            outputs=node.output('Out'))


@op_mapper('fill_constant')
class Constant():
    support_opset_version_range = (1, 12)

    @classmethod
    def opset_1(cls, graph, node, **kw):
        value = node.attr('value')
        dtype = node.attr('dtype')
        shape = node.attr('shape')

        if 'ValueTensor' in node.inputs and len(node.input('ValueTensor')) > 0:
            raise Exception(
                "paddle.full with tensor value parameter is not supported yet.")

        value = np.ones(shape) * value
        value = value.astype(dtypes.DTYPE_PADDLE_NUMPY_MAP[dtype])
        value = value.flatten().tolist()
        if len(shape) == 0 and len(node.input('ShapeTensor')) > 0:
            shape_tensor = mapper_helper.cast(
                graph,
                node.input('ShapeTensor', 0),
                node.input_dtype('ShapeTensor', 0), 'int64')
            graph.make_node(
                'ConstantOfShape',
                inputs=shape_tensor,
                outputs=node.output('Out'),
                attrs={
                    'dims': [1],
                    'dtype': dtypes.DTYPE_PADDLE_ONNX_MAP[dtype],
                    'value': value
                })
        else:
            graph.make_node(
                'Constant',
                inputs=[],
                outputs=node.output('Out'),
                attrs={
                    'dims': shape,
                    'dtype': dtypes.DTYPE_PADDLE_ONNX_MAP[dtype],
                    'value': value
                })


@op_mapper(['lookup_table_v2', 'lookup_table'])
class Embedding():
    support_opset_version_range = (1, 15)

    @classmethod
    def opset_1(cls, graph, node, **kw):
        ids = node.input('Ids', 0)
        if node.type == 'lookup_table' and node.input_shape('Ids', 0)[-1] == 1:
            ids = mapper_helper.squeeze_helper(graph,
                                               node.input('Ids', 0), [-1])
        padding_idx = node.attr('padding_idx')
        input_shape = node.input_shape('W', 0)
        if padding_idx != -1:
            key = node.input('W', 0)
            if -1 in input_shape:
                assert False, "opset version < 11 do not support padding_idx !=-1 and weight is tensor with dynamic shape, please set opset version > 11 or use input_spec to set input shape"
            else:
                data = np.ones(shape=input_shape, dtype=np.float32)
                data[padding_idx] = 0.0
                dtype = dtypes.DTYPE_PADDLE_ONNX_MAP[node.input_dtype('W', 0)]
                constant = graph.make_node(
                    'Constant',
                    dtype=dtype,
                    dims=input_shape,
                    value=data.flatten().tolist())
                weight_node = graph.make_node(
                    'Mul', inputs=[node.input('W', 0), constant])
                graph.make_node(
                    'Gather',
                    inputs=[weight_node, ids],
                    outputs=node.output('Out'))
        else:
            graph.make_node(
                'Gather',
                inputs=[node.input('W', 0), ids],
                outputs=node.output('Out'))

    @classmethod
    def opset_11(cls, graph, node, **kw):
        ids = node.input('Ids', 0)
        if node.type == 'lookup_table' and node.input_shape('Ids', 0)[-1] == 1:
            ids = graph.make_node(
                'Squeeze', inputs=node.input('Ids', 0), axes=[-1])

        padding_idx = node.attr('padding_idx')
        input_shape = node.input_shape('W', 0)
        if padding_idx != -1:
            if -1 in input_shape:
                dtype = dtypes.ONNX.FLOAT
                if node.input_dtype('W', 0) == paddle.float64:
                    dtype = dtypes.ONNX.DOUBLE
                replace_shape = list(copy.copy(input_shape))
                del (replace_shape[0])
                replace_data = constant = graph.make_node(
                    'Constant',
                    dtype=dtype,
                    dims=replace_shape,
                    value=[0.0] * np.prod(replace_shape))
                index = graph.make_node(
                    'Constant', dtype=dtypes.ONNX.INT64, value=[padding_idx])
                Scatter_node = graph.make_node(
                    'ScatterND',
                    inputs=[node.input('W', 0), index, replace_data])
                graph.make_node(
                    'Gather',
                    inputs=[Scatter_node, ids],
                    outputs=node.output('Out'))
            else:
                data = np.ones(shape=input_shape, dtype=np.float32)
                data[padding_idx] = 0.0
                dtype = dtypes.DTYPE_PADDLE_ONNX_MAP[node.input_dtype('W', 0)]
                constant = graph.make_node(
                    'Constant',
                    dtype=dtype,
                    dims=input_shape,
                    value=data.flatten().tolist())
                weight_node = graph.make_node(
                    'Mul', inputs=[node.input('W', 0), constant])
                graph.make_node(
                    'Gather',
                    inputs=[weight_node, ids],
                    outputs=node.output('Out'))
        else:
            graph.make_node(
                'Gather',
                inputs=[node.input('W', 0), ids],
                outputs=node.output('Out'))


@op_mapper('fill_constant_batch_size_like')
class FillConstantBatchSizeLike():
    support_opset_version_range = (9, 12)

    @classmethod
    def opset_10(cls, graph, node, **kw):
        out_shape = node.attr('shape')
        input_dim_idx = node.attr('input_dim_idx')
        output_dim_idx = node.attr('output_dim_idx')

        del out_shape[output_dim_idx]
        out_shape.insert(0, 1)

        dtype = dtypes.DTYPE_PADDLE_ONNX_MAP[node.attr('dtype')]
        value = node.attr('value')
        input_shape = node.input_shape('Input', 0)
        value = int(value)
        constant = graph.make_node(
            'Constant',
            dtype=dtype,
            dims=out_shape,
            value=[value] * np.prod(out_shape))

        shape = graph.make_node('Shape', inputs=node.input('Input'))
        start = graph.make_node(
            'Constant', dtype=dtypes.ONNX.INT64, value=[input_dim_idx])
        end = graph.make_node(
            'Constant', dtype=dtypes.ONNX.INT64, value=[input_dim_idx + 1])
        batch = graph.make_node('Slice', inputs=[shape, start, end])
        repeat = graph.make_node(
            'Constant',
            dtype=dtypes.ONNX.INT64,
            value=[1] * (len(out_shape) - 1))
        repeat = graph.make_node('Concat', inputs=[batch, repeat], axis=-1)
        if output_dim_idx == 0:
            graph.make_node(
                'Tile', inputs=[constant, repeat], outputs=node.output('Out'))
        else:
            out = graph.make_node('Tile', inputs=[constant, repeat])
            perm = list(range(len(out_shape)))
            perm[0] = output_dim_idx
            perm[output_dim_idx] = 0
            graph.make_node(
                'Transpose',
                inputs=[out],
                perm=perm,
                outputs=node.output('Out'))


@op_mapper('fill_any_like')
class FullLike():
    '''
    fill_any_like is kernel for paddle op::full_like & ones_like
    '''
    support_opset_version_range = (9, 15)

    @classmethod
    def opset_9(cls, graph, node, **kw):
        shape_node = graph.make_node('Shape', inputs=node.input('X'))
        value = node.attr('value')
        dtype = node.attr('dtype')
        input_dtype = node.input_var('X', 0).dtype
        if dtype is None:
            dtype = input_dtype
        np_dtype = dtypes.DTYPE_PADDLE_STR_MAP[dtype]
        onnx_dtype = dtypes.DTYPE_PADDLE_ONNX_MAP[dtype]
        graph.make_node(
            'ConstantOfShape',
            inputs=[shape_node],
            outputs=node.output('Out'),
            dims=[1],
            dtype=onnx_dtype,
            value=np.array(value).astype(np_dtype))


@op_mapper('fill_zeros_like')
class FullZeroLike():
    '''
    fill_zeros_like is kernel for paddle op::zeros_like
    '''
    support_opset_version_range = (9, 15)

    @classmethod
    def opset_9(cls, graph, node, **kw):
        shape_node = graph.make_node('Shape', inputs=node.input('X'))
        value = 0
        dtype = node.attr('dtype')
        input_dtype = node.input_var('X', 0).dtype
        if dtype is None:
            dtype = input_dtype
        np_dtype = dtypes.DTYPE_PADDLE_STR_MAP[dtype]
        onnx_dtype = dtypes.DTYPE_PADDLE_ONNX_MAP[dtype]
        graph.make_node(
            'ConstantOfShape',
            inputs=[shape_node],
            outputs=node.output('Out'),
            dims=[1],
            dtype=onnx_dtype,
            value=np.array(value).astype(np_dtype))


@op_mapper('gather_nd')
class Gather_nd():
    support_opset_version_range = (11, 15)

    @classmethod
    def opset_11(cls, graph, node, **kw):
        data = node.input('X', 0)
        index = node.input('Index', 0)
        index_dtype = node.input_dtype('Index', 0)
        index_node = None
        if index_dtype != paddle.int64:
            index_node = graph.make_node(
                'Cast', inputs=[node.input('Index', 0)], to=dtypes.ONNX.INT64)
        else:
            index_node = index
        graph.make_node(
            'GatherND', inputs=[data, index_node], outputs=node.output('Out'))


@op_mapper('gather')
class Gather():
    support_opset_version_range = (7, 15)

    @classmethod
    def opset_1(cls, graph, node, **kw):
        axis = node.attr('axis')
        if node.input('Axis', 0) != None:
            axis_node = node.input('Axis', 0)
            # When axis is tensor, only int32 and int64 are supported
            if axis_node not in graph.parameters:
                raise Exception(
                    "Currently does not support the axis parameter as input tensor!"
                )
            else:
                axis = graph.parameters[axis_node].attribute[0].t.int32_data
                if axis is None or len(axis) < 1:
                    axis = graph.parameters[axis_node].attribute[
                        0].t.int64_data[0]
        if axis is None:
            axis = 0
        if len(node.input_shape('Index', 0)) == 1:
            # gather
            graph.make_node(
                'Gather',
                inputs=[node.input('X', 0), node.input('Index', 0)],
                outputs=node.output('Out'),
                attrs={'axis': axis})
        else:
            raise Exception(
                "please try to convert OP:gather(indices's rank >1) with opset_version >= 11."
            )

    @classmethod
    def opset_11(cls, graph, node, **kw):
        axis = node.attr('axis')
        if node.input('Axis', 0) != None:
            axis_node = node.input('Axis', 0)
            # When axis is tensor, only int32 and int64 are supported
            if axis_node not in graph.parameters:
                raise Exception(
                    "Currently does not support the axis parameter as input tensor!"
                )
            else:
                axis = graph.parameters[axis_node].attribute[0].t.int32_data
                if axis is None or len(axis) < 1:
                    axis = graph.parameters[axis_node].attribute[
                        0].t.int64_data[0]
        if axis is None:
            axis = 0
        if len(node.input_shape('Index', 0)) == 1:
            # gather
            graph.make_node(
                'Gather',
                inputs=[node.input('X', 0), node.input('Index', 0)],
                outputs=node.output('Out'),
                attrs={'axis': axis})
        else:
            # gather_nd
            index_dtype = node.input_dtype('Index', 0)
            if index_dtype != paddle.int64:
                index_node = graph.make_node(
                    'Cast',
                    inputs=[node.input('Index', 0)],
                    to=dtypes.ONNX.INT64)
                graph.make_node(
                    'GatherND',
                    inputs=[node.input('X', 0), index_node],
                    outputs=node.output('Out'))
            else:
                graph.make_node(
                    'GatherND',
                    inputs=[node.input('X', 0), node.input('Index', 0)],
                    outputs=node.output('Out'))


@op_mapper('squeeze2')
class Squeeze():
    support_opset_version_range = (1, 12)

    @classmethod
    def opset_1(cls, graph, node, **kw):
        axes = cls.compute_axes(node)
        axes.sort()
        graph.make_node(
            'Squeeze',
            inputs=[node.input('X', 0)],
            outputs=node.output('Out'),
            axes=axes)

    @classmethod
    def opset_13(cls, graph, node, **kw):
        axes = cls.compute_axes(node)
        axes_node = graph.make_node(
            'Constant', attrs={'dtype': dtypes.ONNX.INT64,
                               'value': axes})
        graph.make_node(
            'Squeeze',
            inputs=[node.input('X', 0)] + [axes_node],
            outputs=node.output('Out'))

    @classmethod
    def compute_axes(cls, node):
        axes = node.attr('axes')
        input_x = node.input('X')[0]
        ndim = node.block.vars[input_x].ndim
        shape = node.block.vars[input_x].shape
        if len(axes) == 0:
            axes = [i for i, axis in enumerate(shape) if axis == 1]
            assert len(
                axes
            ) > 0, "axes response to input data shape should at least have 1."
        else:
            axes = [
                axis + ndim if axis < 0 else axis for i, axis in enumerate(axes)
            ]
        return axes


@op_mapper('assign_value')
class Assign():
    support_opset_version_range = (1, 12)

    @classmethod
    def opset_1(cls, graph, node, **kw):
        if len(node.input_names) > 0:
            graph.make_node(
                'Identity', inputs=node.input('X'), outputs=node.output('Out'))
        else:
            parameters = {}
            value = np.array(node.attr('fp32_values'))
            if value is None or value.size < 1:
                value = np.array(node.attr('int32_values'))
            if value is None or value.size < 1:
                value = np.array(node.attr('int64_values'))
            parameter = {
                'data': value,
                'dtype': node.attr('dtype'),
                'shape': node.attr('shape')
            }
            parameters[node.output('Out', 0)] = parameter
            graph.build_parameters(parameters)


@op_mapper('transpose2')
class Transpose():
    support_opset_version_range = (7, 15)

    @classmethod
    def opset_7(cls, graph, node, **kw):
        graph.make_node(
            'Transpose',
            inputs=node.input('X'),
            outputs=node.output('Out'),
            perm=node.attr('axis'))


@op_mapper('flatten2')
class Flatten():
    support_opset_version_range = (1, 12)

    @classmethod
    def opset_1(cls, graph, node, **kw):
        graph.make_node(
            'Flatten',
            inputs=node.input('X'),
            outputs=node.output('Out'),
            axis=node.attr('axis'))


@op_mapper('flatten_contiguous_range')
class FlattenContiguousRange():
    support_opset_version_range = (7, 15)

    @classmethod
    def opset_7(cls, graph, node, **kw):
        dims = len(node.input_shape('X', 0))
        start_axis = node.attr('start_axis')
        end_axis = node.attr('stop_axis')
        shape_node = graph.make_node('Shape', inputs=node.input('X'))
        if end_axis < dims - 1:
            slice1 = mapper_helper.slice_helper(
                graph, shape_node, axes=[0], starts=[0], ends=[start_axis])
            slice3 = mapper_helper.slice_helper(
                graph, shape_node, axes=[0], starts=[end_axis + 1],
                ends=[dims])
            slices = [
                slice1, graph.make_node(
                    'Constant', value=[-1], dtype=dtypes.ONNX.INT64), slice3
            ]
        else:
            slice1 = mapper_helper.slice_helper(
                graph, shape_node, axes=[0], starts=[0], ends=[start_axis])
            slices = [
                slice1, graph.make_node(
                    'Constant', value=[-1], dtype=dtypes.ONNX.INT64)
            ]
        final_shape = graph.make_node('Concat', inputs=slices, axis=0)
        graph.make_node(
            'Reshape',
            inputs=[node.input('X')[0], final_shape],
            outputs=node.output('Out'))


@op_mapper('reshape2')
class Reshape():
    support_opset_version_range = (7, 15)

    @classmethod
    def opset_7(cls, graph, node, **kw):
        shape_name = 'ShapeTensor'
        if shape_name not in node.inputs or len(node.input(shape_name)) == 0:
            shape_name = 'Shape'
        if shape_name not in node.inputs or len(node.input(shape_name)) == 0:
            if node.attr('shape') is None or len(node.attr('shape')) == 0:
                raise Exception("shape tensor and shape attrubite all unkown.")
        if len(node.input(shape_name)) > 1:
            dims = []
            for i in range(len(node.input(shape_name))):
                dim = node.input(shape_name)[i]
                dim = graph.make_node(
                    'Cast', inputs=[dim], to=dtypes.ONNX.INT64)
                dims.append(dim)
            shape = graph.make_node('Concat', inputs=dims, axis=-1)
            graph.make_node(
                'Reshape',
                inputs=[node.input('X')[0], shape],
                outputs=node.output('Out'))
        elif len(node.input(shape_name)) == 1:
            cast_shape_node = graph.make_node(
                'Cast', inputs=node.input(shape_name), to=dtypes.ONNX.INT64)
            graph.make_node(
                'Reshape',
                inputs=[node.input('X')[0], cast_shape_node],
                outputs=node.output('Out'))
        elif node.attr('shape') is not None and len(node.attr('shape')) > 0:
            shape_node = graph.make_node(
                'Constant',
                attrs={
                    'dtype': dtypes.ONNX.INT64,
                    'value': node.attr('shape')
                })
            reshape_node = graph.make_node(
                'Reshape',
                inputs=[node.input('X')[0], shape_node],
                outputs=node.output('Out'))


@op_mapper('unsqueeze2')
class Unsqueeze():
    support_opset_version_range = (1, 12)

    @classmethod
    def opset_1(cls, graph, node, **kw):
        axes = cls.get_axes(graph, node)
        graph.make_node(
            'Unsqueeze',
            inputs=node.input('X'),
            outputs=node.output('Out'),
            axes=axes)

    @classmethod
    def opset_13(cls, graph, node, **kw):
        axes_node = cls.get_axes(graph, node, return_node=True)
        graph.make_node(
            'Unsqueeze',
            inputs=node.input('X') + [axes_node],
            outputs=node.output('Out'))

    @classmethod
    def get_axes(cls, graph, node, return_node=False):
        axes_node = None
        ndim = node.block.vars[node.input('X')[0]].ndim
        if len(node.attr('axes')) > 0:
            axes = node.attr('axes')
        else:
            axes_node = node.input('AxesTensor')[0]
            axes = graph.parameters[axes_node].attribute[0].t.int64_data
        # axes is list of non-negative integers
        axes = [
            axis + ndim + i + 1 if axis < 0 else axis
            for i, axis in enumerate(axes)
        ]

        axes_copy = axes.copy()
        assert sorted(
            axes) == axes_copy, "axes must be arranged in the following order"
        assert len(set(axes)) == len(axes), "axes have duplicate axis"

        if return_node:
            if axes_node is None:
                axes_node = graph.make_node(
                    'Constant',
                    attrs={'dtype': dtypes.ONNX.INT64,
                           'value': axes})
            return axes_node
        return axes


@op_mapper('reciprocal')
class Reciprocal():
    support_opset_version_range = (7, 15)

    @classmethod
    def opset_1(cls, graph, node, **kw):
        graph.make_node(
            'Reciprocal', inputs=node.input('X'), outputs=node.output('Out'))


@op_mapper('cast')
class Cast():
    support_opset_version_range = (7, 15)

    @classmethod
    def opset_1(cls, graph, node, **kw):
        graph.make_node(
            'Cast',
            inputs=node.input('X'),
            outputs=node.output('Out'),
            to=dtypes.DTYPE_PADDLE_ONNX_MAP[node.attr('out_dtype')])


@op_mapper('linspace')
class Linspace():
    support_opset_version_range = (9, 15)

    @classmethod
    def opset_9(cls, graph, node, **kw):
        start = node.input('Start', 0)
        stop = node.input('Stop', 0)
        num = node.input('Num', 0)
        dtype = node.attr('dtype')

        start = graph.make_node('Cast', inputs=[start], to=dtypes.ONNX.FLOAT)
        stop = graph.make_node('Cast', inputs=[stop], to=dtypes.ONNX.FLOAT)

        sub_a_node = graph.make_node('Sub', inputs=[stop, start])

        one_node = graph.make_node(
            'Constant',
            dtype=dtypes.DTYPE_PADDLE_ONNX_MAP[node.input_dtype('Num', 0)],
            value=[1])

        sub_b_node = graph.make_node('Sub', inputs=[num, one_node])

        sub_b_float_node = graph.make_node(
            'Cast', inputs=[sub_b_node], to=dtypes.ONNX.FLOAT)

        step = graph.make_node('Div', inputs=[sub_a_node, sub_b_float_node])

        range_tensor = graph.make_node(
            'Cast', inputs=[num], to=dtypes.ONNX.INT64)

        one_like_node = graph.make_node(
            'ConstantOfShape',
            inputs=[range_tensor],
            dtype=dtypes.ONNX.FLOAT,
            value=[1])

        none_zero_node = graph.make_node('NonZero', inputs=[one_like_node])

        trans_none_zero_node = graph.make_node(
            'Transpose', inputs=[none_zero_node], perm=[1, 0])

        trans_squeeze = mapper_helper.squeeze_helper(graph,
                                                     trans_none_zero_node, [1])

        trans_squeeze = graph.make_node(
            'Cast', inputs=[trans_squeeze], to=dtypes.ONNX.FLOAT)

        mul_node = graph.make_node('Mul', inputs=[trans_squeeze, step])

        add_node = graph.make_node('Add', inputs=[mul_node, start])
        graph.make_node(
            'Cast',
            inputs=[add_node],
            outputs=node.output('Out'),
            to=dtypes.DTYPE_PADDLE_ONNX_MAP[node.input_dtype('Start', 0)])


@op_mapper('clip')
class Clip():
    support_opset_version_range = (7, 15)

    @classmethod
    def opset_1(cls, graph, node, **kw):
        min_value = node.attr('min')
        max_value = node.attr('max')
        x_dtype = node.input_dtype('X', 0)
        if node.input('Max', 0) is None or len(node.input('Max')) == 0:
            max_ = max_value
        else:
            max_ = node.input('Max', 0)
        if node.input('Min', 0) is None or len(node.input('Min')) == 0:
            min_ = min_value
        else:
            min_ = node.input('Min', 0)
        mapper_helper.clip_helper(graph,
                                  node.input('X', 0), max_, min_,
                                  node.output('Out', 0), x_dtype)


@op_mapper(['pad2d', 'pad3d'])
class Pad():
    support_opset_version_range = (1, 12)

    @classmethod
    def opset_1(cls, graph, node, **kw):
        if node.attr('mode') == 'replicate':
            mode = 'edge'
        elif node.attr('mode') == 'circular':
            raise Exception("The padding mode = circular is not supported, " \
                            "Please try the other three ways")
        else:
            mode = node.attr('mode')
        pads = cls.convert_padding(node, **kw)
        if pads is None:
            key = node.input('Paddings', 0)
            padding = None
            if key in graph.parameters.keys():
                paddings = graph.parameters[key].attribute[0].t.int32_data
                if node.attr('data_format') == 'NCHW':
                    pads = [
                        0, 0, paddings[0], paddings[2], 0, 0, paddings[1],
                        paddings[3]
                    ]
                elif node.attr('data_format') == 'NHWC':
                    pads = [
                        0, paddings[0], paddings[2], 0, 0, paddings[1],
                        paddings[3], 0
                    ]
                elif node.attr('data_format') == 'NCDHW':
                    pads = [
                        0, 0, paddings[4], paddings[2], paddings[0], 0, 0,
                        paddings[5], paddings[3], paddings[1]
                    ]
                elif node.attr('data_format') == 'NDHWC':
                    pads = [
                        0, paddings[4], paddings[2], paddings[0], 0, 0,
                        paddings[5], paddings[3], paddings[1], 0
                    ]
            else:
                raise Exception("In Pad op, padding can not be tensor" \
                            "Please set opset version >= 11")

        value = None
        if node.attr('pad_value') is not None:
            value = node.attr('pad_value')
        elif node.attr('value') is not None:
            value = node.attr('value')
        graph.make_node(
            'Pad',
            inputs=node.input('X'),
            outputs=node.output('Out'),
            mode=mode,
            value=value,
            pads=pads)

    @classmethod
    def opset_11(cls, graph, node, **kw):
        pads = cls.convert_padding(node, **kw)
        if node.attr('mode') == 'replicate':
            mode = 'edge'
        elif node.attr('mode') == 'circular':
            raise Exception("The padding mode = circular is not supported, " \
                            "Please try the other three ways")
        else:
            mode = node.attr('mode')
        pads_node = None
        if isinstance(pads, list):
            pads_node = graph.make_node(
                'Constant', attrs={'dtype': dtypes.ONNX.INT64,
                                   'value': pads})
        else:
            key = node.input('Paddings', 0)
            padding = None
            if key in graph.parameters.keys():
                paddings = graph.parameters[key].attribute[0].t.int32_data
                onnx_paddings = None
                if node.attr('data_format') == 'NCHW':
                    onnx_paddings = [
                        0, 0, paddings[0], paddings[2], 0, 0, paddings[1],
                        paddings[3]
                    ]
                elif node.attr('data_format') == 'NHWC':
                    onnx_paddings = [
                        0, paddings[0], paddings[2], 0, 0, paddings[1],
                        paddings[3], 0
                    ]
                elif node.attr('data_format') == 'NCDHW':
                    onnx_paddings = [
                        0, 0, paddings[4], paddings[2], paddings[0], 0, 0,
                        paddings[5], paddings[3], paddings[1]
                    ]
                elif node.attr('data_format') == 'NDHWC':
                    onnx_paddings = [
                        0, paddings[4], paddings[2], paddings[0], 0, 0,
                        paddings[5], paddings[3], paddings[1], 0
                    ]

                pads_node = graph.make_node(
                    'Constant',
                    attrs={'dtype': dtypes.ONNX.INT64,
                           'value': onnx_paddings})
            else:
                padding_node = node.input('Paddings', 0)
                casted_padding_node = graph.make_node(
                    'Cast', inputs=[padding_node], to=dtypes.ONNX.FLOAT)
                zero_node = None
                if node.attr('data_format') == 'NCHW' or node.attr(
                        'data_format') == 'NHWC':
                    zero_node = graph.make_node(
                        'Constant', dtype=dtypes.ONNX.FLOAT, value=[0] * 8)
                else:
                    zero_node = graph.make_node(
                        'Constant', dtype=dtypes.ONNX.FLOAT, value=[0] * 10)
                index = None
                if node.attr('data_format') == 'NCHW':
                    index = graph.make_node(
                        'Constant', dtype=dtypes.ONNX.INT32,
                        value=[2, 6, 3, 7])
                elif node.attr('data_format') == 'NHWC':
                    index = graph.make_node(
                        'Constant', dtype=dtypes.ONNX.INT32,
                        value=[1, 5, 2, 6])
                elif node.attr('data_format') == 'NCDHW':
                    index = graph.make_node(
                        'Constant',
                        dtype=dtypes.ONNX.INT32,
                        value=[4, 9, 3, 8, 2, 7])
                elif node.attr('data_format') == 'NDHWC':
                    index = graph.make_node(
                        'Constant',
                        dtype=dtypes.ONNX.INT32,
                        value=[3, 8, 2, 7, 1, 6])

                float_paddle_node = graph.make_node(
                    'ScatterElements',
                    inputs=[zero_node, index, casted_padding_node])
                paddle_node = graph.make_node(
                    'Cast', inputs=[float_paddle_node], to=dtypes.ONNX.INT64)
                pads_node = paddle_node

        value = None
        if node.attr('pad_value') is not None:
            value = node.attr('pad_value')
        elif node.attr('value') is not None:
            value = node.attr('value')
        value_node = graph.make_node(
            'Constant',
            attrs={
                'dtype': dtypes.DTYPE_PADDLE_ONNX_MAP[node.input_dtype('X', 0)],
                'value': value
            })

        graph.make_node(
            'Pad',
            inputs=node.input('X') + [pads_node, value_node],
            outputs=node.output('Out'),
            mode=mode)

    @classmethod
    def convert_padding(cls, node, **kw):
        x_shape = node.input_shape('X', 0)
        paddings = node.attr('paddings')
        if paddings == []:
            return None
        onnx_paddings = None
        if node.attr('data_format') == 'NCHW':
            onnx_paddings = [
                0, 0, paddings[0], paddings[2], 0, 0, paddings[1], paddings[3]
            ]
        elif node.attr('data_format') == 'NHWC':
            onnx_paddings = [
                0, paddings[0], paddings[2], 0, 0, paddings[1], paddings[3], 0
            ]
        elif node.attr('data_format') == 'NCDHW':
            onnx_paddings = [
                0, 0, paddings[4], paddings[2], paddings[0], 0, 0, paddings[5],
                paddings[3], paddings[1]
            ]
        elif node.attr('data_format') == 'NDHWC':
            onnx_paddings = [
                0, paddings[4], paddings[2], paddings[0], 0, 0, paddings[5],
                paddings[3], paddings[1], 0
            ]
        return onnx_paddings


@op_mapper('gaussian_random')
class GaussianRandom():
    support_opset_version_range = (7, 15)

    @classmethod
    def opset_7(cls, graph, node, **kw):
        shape_input_list = node.input('ShapeTensorList')
        shape_input = None
        if len(shape_input_list) == 0:
            shape_input = node.input('ShapeTensor')
        else:
            shape_input = graph.make_node(
                "Concat", inputs=node.input('ShapeTensorList'), axis=0)
        if shape_input is None or len(shape_input) == 0:
            graph.make_node(
                'RandomNormal',
                dtype=dtypes.DTYPE_PADDLE_ONNX_MAP[node.attr('dtype')],
                outputs=node.output('Out'),
                shape=node.attr('shape'),
                seed=float(node.attr('seed')),
                mean=node.attr('mean'),
                scale=node.attr('std'))
        else:
            cast_input_shape = graph.make_node(
                'Cast', inputs=shape_input, to=dtypes.ONNX.INT64)
            zero_like_node = graph.make_node(
                'ConstantOfShape',
                inputs=cast_input_shape,
                dims=[1],
                dtype=dtypes.ONNX.FLOAT,
                value=[0])
            graph.make_node(
                'RandomNormalLike',
                dtype=dtypes.DTYPE_PADDLE_ONNX_MAP[node.attr('dtype')],
                outputs=node.output('Out'),
                inputs=zero_like_node,
                seed=float(node.attr('seed')),
                mean=node.attr('mean'),
                scale=node.attr('std'))


@op_mapper('uniform_random_batch_size_like')
class UniformRandom():
    support_opset_version_range = (1, 12)

    @classmethod
    def opset_1(cls, graph, node, **kw):
        graph.make_node(
            'RandomUniformLike',
            inputs=node.input('Input'),
            outputs=node.output('Out'),
            high=node.attr('max'),
            dtype=dtypes.DTYPE_PADDLE_ONNX_MAP[node.attr('dtype')],
            low=node.attr('min'),
            seed=float(node.attr('seed')), )


@op_mapper('uniform_random')
class UniformRandom():
    support_opset_version_range = (7, 15)

    @classmethod
    def opset_7(cls, graph, node, **kw):
        shape_input_list = node.input('ShapeTensorList')
        shape_input = None
        if len(shape_input_list) == 0:
            shape_input = node.input('ShapeTensor')
        else:
            shape_input = graph.make_node(
                "Concat", inputs=node.input('ShapeTensorList'), axis=0)
        if shape_input is None or len(shape_input) == 0:
            graph.make_node(
                'RandomUniform',
                dtype=dtypes.DTYPE_PADDLE_ONNX_MAP[node.attr('dtype')],
                outputs=node.output('Out'),
                shape=node.attr('shape'),
                seed=float(node.attr('seed')),
                low=node.attr('min'),
                high=node.attr('max'))
        else:
            cast_input_shape = graph.make_node(
                'Cast', inputs=shape_input, to=dtypes.ONNX.INT64)
            zero_like_node = graph.make_node(
                'ConstantOfShape',
                inputs=cast_input_shape,
                dtype=dtypes.ONNX.FLOAT,
                value=[0])
            graph.make_node(
                'RandomUniformLike',
                dtype=dtypes.DTYPE_PADDLE_ONNX_MAP[node.attr('dtype')],
                outputs=node.output('Out'),
                inputs=zero_like_node,
                seed=float(node.attr('seed')),
                low=node.attr('min'),
                high=node.attr('max'))


@op_mapper(
    [
        'bilinear_interp', 'nearest_interp', 'bilinear_interp_v2',
        'nearest_interp_v2', 'bicubic_interp_v2'
    ],
    mapper_dict={
        'bilinear_interp': 'linear',
        'nearest_interp': 'nearest',
        'bilinear_interp_v2': 'linear',
        'nearest_interp_v2': 'nearest',
        'bicubic_interp_v2': 'cubic'
    })
class Resize():
    support_opset_version_range = (9, 12)

    @classmethod
    def opset_9(cls, graph, node, **kw):
        inputs = [node.input('X')[0]]
        resize_type = kw['mapper_dict'][node.type]
        if node.attr('align_corners') or node.attr('align_mode') == 0:
            raise Exception(
                "Resize in onnx(opset<=10) only support coordinate_transformation_mode: " \
                "'asymmetric', Try converting with opset_version 11"
            )
        if len(node.input('OutSize')) > 0 or len(node.input('SizeTensor')) > 0:
            in_shape, out_shape = cls.compute_output_shape(
                graph, node, node.input('X')[0], opset_version=9)
            cast_shape_node2 = graph.make_node(
                'Cast', inputs=[out_shape], to=dtypes.ONNX.FLOAT)
            cast_shape_node0 = graph.make_node(
                'Cast', inputs=[in_shape], to=dtypes.ONNX.FLOAT)
            node_h_w_scales = graph.make_node(
                'Div', inputs=[cast_shape_node2, cast_shape_node0])
            inputs.append(node_h_w_scales)
        elif 'Scale' in node.inputs and len(node.input('Scale')) > 0:
            scale = node.input('Scale')[0]
            inputs.append(out_shape)
        else:
            out_shape = [node.attr('out_h'), node.attr('out_w')]
            scale = node.attr('scale')
            if isinstance(scale, (tuple, list)):
                scale_h = scale[0]
                scale_w = scale[1]
            else:
                scale_h = scale
                scale_w = scale
            if out_shape.count(-1) > 0:
                scale_node = graph.make_node(
                    'Constant',
                    attrs={
                        'dtype': dtypes.ONNX.FLOAT,
                        'value': [1, 1, scale_h, scale_w]
                    })
                inputs.append(scale_node)
            else:
                raise Exception("Unexpected situation happend")
        graph.make_node(
            'Upsample',
            inputs=inputs,
            outputs=node.output('Out'),
            mode=resize_type)

    @classmethod
    def opset_10(cls, graph, node, **kw):
        inputs = [node.input('X')[0]]
        resize_type = kw['mapper_dict'][node.type]
        if node.attr('align_corners') or node.attr('align_mode') == 0:
            raise Exception(
                "Resize in onnx(opset<=10) only support coordinate_transformation_mode:" \
                " 'asymmetric', Try converting with opset_version 11"
            )
        if len(node.input('OutSize')) > 0 or len(node.input('SizeTensor')) > 0:
            in_shape, out_shape = cls.compute_output_shape(graph, node,
                                                           node.input('X')[0])
            cast_shape_node2 = graph.make_node(
                'Cast', inputs=[out_shape], to=dtypes.ONNX.FLOAT)
            cast_shape_node0 = graph.make_node(
                'Cast', inputs=[in_shape], to=dtypes.ONNX.FLOAT)
            node_h_w_scales = graph.make_node(
                'Div', inputs=[cast_shape_node2, cast_shape_node0])
            inputs.append(node_h_w_scales)
        elif 'Scale' in node.inputs and len(node.input('Scale')) > 0:
            scale = node.input('Scale')[0]
            inputs.append(scale)
        else:
            out_shape = [node.attr('out_h'), node.attr('out_w')]
            scale = node.attr('scale')
            if isinstance(scale, float):
                scale = [1, 1, scale, scale]
            else:
                scale = [1, 1] + scale
            if out_shape.count(-1) > 0:
                scale_node = graph.make_node(
                    'Constant',
                    attrs={'dtype': dtypes.ONNX.FLOAT,
                           'value': scale})
                inputs.append(scale_node)
            else:
                raise Exception("Unexpected situation happend")
        graph.make_node(
            'Resize',
            inputs=inputs,
            outputs=node.output('Out'),
            mode=resize_type)

    @classmethod
    def opset_11(cls, graph, node, **kw):
        date_layout = node.attr('data_layout')
        dim = len(node.input_shape('X', 0))
        if date_layout == 'NHWC':
            if dim == 4:
                perm = [0, 3, 1, 2]
                perm_t = [0, 2, 3, 1]
            elif dim == 5:
                perm = [0, 4, 1, 2, 3]
                perm_t = [0, 2, 3, 4, 1]
            input = graph.make_node(
                'Transpose', inputs=node.input('X')[0], perm=perm)
        else:
            input = node.input('X')[0]
        node_lists = []
        resize_type = kw['mapper_dict'][node.type]
        coordinate_transformation_mode = ''
        if node.attr('align_corners'):
            coordinate_transformation_mode = 'align_corners'
        elif node.type == 'nearest_interp':
            coordinate_transformation_mode = 'half_pixel'
        else:
            if node.attr('align_mode') == 1:
                coordinate_transformation_mode = 'asymmetric'
            else:
                coordinate_transformation_mode = 'half_pixel'
        if node.type == 'nearest_interp_v2':
            coordinate_transformation_mode = 'asymmetric'
        roi_node = graph.make_node(
            'Constant',
            attrs={
                'dtype': dtypes.ONNX.FLOAT,
                'value': [1, 1, 1, 1, 1, 1, 1, 1]
            })
        inputs = [input, roi_node]
        node_lists.append(roi_node)

        out_size = node.input('OutSize')
        size_tensor = node.input('SizeTensor')
        scale = node.input('Scale')
        if (out_size is not None and len(out_size) > 0) or (
                size_tensor is not None and len(size_tensor) > 0):
            empty_node = graph.make_node(
                'Constant', attrs={'dtype': dtypes.ONNX.FLOAT,
                                   'value': []})
            inputs.append(empty_node)
            _, out_shape = cls.compute_output_shape(graph, node, input)
            inputs.append(out_shape)
        elif scale is not None and len(scale) > 0:
            scale = node.input('Scale')[0]
            inputs.append(scale)
        else:
            if dim == 4:
                out_shape = [node.attr('out_h'), node.attr('out_w')]
            else:
                out_shape = [
                    node.attr('out_d'), node.attr('out_h'), node.attr('out_w')
                ]

            scale = node.attr('scale')
            if isinstance(scale, float):
                scale = [1, 1, scale, scale]
            else:
                scale = [1, 1] + scale

            if out_shape.count(-1) > 0:
                scale_node = graph.make_node(
                    'Constant',
                    attrs={'dtype': dtypes.ONNX.FLOAT,
                           'value': scale})
                inputs.append(scale_node)
            else:
                empty_node = graph.make_node(
                    'Constant',
                    attrs={'dtype': dtypes.ONNX.FLOAT,
                           'value': []})
                in_shape, out_shape = cls.compute_output_shape_by_size(
                    graph, node, input, dim)
                inputs += [empty_node, out_shape]
        if date_layout == 'NHWC':
            if resize_type == 'nearest' and coordinate_transformation_mode == 'asymmetric':
                out_node = graph.make_node(
                    'Resize',
                    inputs=inputs,
                    mode=resize_type,
                    coordinate_transformation_mode=coordinate_transformation_mode,
                    nearest_mode='floor')
            else:
                out_node = graph.make_node(
                    'Resize',
                    inputs=inputs,
                    mode=resize_type,
                    coordinate_transformation_mode=coordinate_transformation_mode
                )
            graph.make_node(
                'Transpose',
                inputs=out_node,
                perm=perm_t,
                outputs=node.output('Out'))
        else:
            if resize_type == 'nearest' and coordinate_transformation_mode == 'asymmetric':
                graph.make_node(
                    'Resize',
                    inputs=inputs,
                    outputs=node.output('Out'),
                    mode=resize_type,
                    coordinate_transformation_mode=coordinate_transformation_mode,
                    nearest_mode='floor')
            else:
                graph.make_node(
                    'Resize',
                    inputs=inputs,
                    outputs=node.output('Out'),
                    mode=resize_type,
                    coordinate_transformation_mode=coordinate_transformation_mode
                )

    @classmethod
    def compute_output_shape(cls, graph, node, input, opset_version=10):
        shape_node0 = graph.make_node('Shape', inputs=input)
        if opset_version < 10:
            shape_node1 = graph.make_node(
                'Slice', inputs=[shape_node0], starts=[0], ends=[2])
        else:
            starts_node = graph.make_node(
                'Constant', attrs={'dtype': dtypes.ONNX.INT64,
                                   'value': [0]})
            ends_node = graph.make_node(
                'Constant', attrs={'dtype': dtypes.ONNX.INT64,
                                   'value': [2]})
            shape_node1 = graph.make_node(
                'Slice', inputs=[shape_node0, starts_node, ends_node])
        if len(node.input('OutSize')) > 0:
            cast_shape_node = graph.make_node(
                'Cast', inputs=node.input('OutSize'), to=dtypes.ONNX.INT64)
        else:
            concat_shape_node = graph.make_node(
                "Concat", inputs=node.input('SizeTensor'), axis=0)
            cast_shape_node = graph.make_node(
                'Cast', inputs=[concat_shape_node], to=dtypes.ONNX.INT64)
        shape_node2 = graph.make_node(
            'Concat', inputs=[shape_node1, cast_shape_node], axis=0)
        return shape_node0, shape_node2

    @classmethod
    def compute_output_shape_by_size(cls,
                                     graph,
                                     node,
                                     input,
                                     dim,
                                     opset_version=10):
        shape_node0 = graph.make_node('Shape', inputs=input)
        if opset_version < 10:
            shape_node1 = graph.make_node(
                'Slice', inputs=[shape_node0], starts=[0], ends=[2])
        else:
            starts_node = graph.make_node(
                'Constant', attrs={'dtype': dtypes.ONNX.INT64,
                                   'value': [0]})
            ends_node = graph.make_node(
                'Constant', attrs={'dtype': dtypes.ONNX.INT64,
                                   'value': [2]})
            shape_node1 = graph.make_node(
                'Slice', inputs=[shape_node0, starts_node, ends_node])
        if dim == 4:
            out_shape = [node.attr('out_h'), node.attr('out_w')]
        else:
            out_shape = [
                node.attr('out_d'), node.attr('out_h'), node.attr('out_w')
            ]
        shape_node2 = graph.make_node(
            'Constant', attrs={'dtype': dtypes.ONNX.INT64,
                               'value': out_shape})
        shape_node3 = graph.make_node(
            'Concat', inputs=[shape_node1, shape_node2], axis=0)
        return shape_node0, shape_node3


@op_mapper('pixel_shuffle')
class PixelShuffle():
    support_opset_version_range = (11, 15)

    @classmethod
    def opset_11(cls, graph, node, **kw):
        upscale_factor = node.attr('upscale_factor')

        node = graph.make_node(
            'DepthToSpace',
            inputs=node.input('X'),
            outputs=node.output('Out'),
            blocksize=upscale_factor,
            mode='CRD')


@op_mapper('scatter')
class Scatter():
    support_opset_version_range = (11, 15)

    @classmethod
    def opset_11(cls, graph, node, **kw):
        shape = graph.make_node(
            'Constant',
            value=[node.input_shape('Ids', 0)[0], 1],
            dtype=dtypes.ONNX.INT64)
        reshape_index = graph.make_node(
            'Reshape', inputs=[node.input('Ids', 0), shape])
        if not node.attr('overwrite'):
            raise Exception("overwrite = False not support yet.")
        else:
            graph.make_node(
                'ScatterND',
                inputs=[
                    node.input('X', 0), reshape_index, node.input('Updates', 0)
                ],
                outputs=node.output('Out'))


@op_mapper('scatter_nd_add')
class ScatterndAdd():
    support_opset_version_range = (11, 12)

    @classmethod
    def opset_11(cls, graph, node, **kw):
        shape = graph.make_node('Shape', inputs=node.input('X', 0))
        zero_like_node = graph.make_node(
            'ConstantOfShape',
            inputs=[shape],
            dims=[1],
            dtype=dtypes.ONNX.FLOAT,
            value=[0])
        add_node = graph.make_node(
            'ScatterND',
            inputs=[
                zero_like_node, node.input('Index', 0), node.input('Updates', 0)
            ], )
        graph.make_node(
            'Add',
            inputs=[node.input('X', 0), add_node],
            outputs=node.output('Out'))


@op_mapper('meshgrid')
class Meshgrid():
    support_opset_version_range = (8, 15)

    @classmethod
    def opset_8(cls, graph, node, **kw):
        tensors = [t for t in list(node.input('X'))]
        tensors_shape = [graph.make_node('Shape', inputs=t) for t in tensors]
        out_shape = graph.make_node('Concat', inputs=tensors_shape, axis=0)
        out = []
        for i, t in enumerate(tensors):
            shape_i = [
                graph.make_node(
                    'Constant',
                    attrs={'dtype': dtypes.ONNX.INT64,
                           'value': [1]})
            ] * len(tensors)
            shape_i[i] = tensors_shape[i]
            t_reshaped = graph.make_node(
                'Reshape',
                inputs=[t, graph.make_node(
                    'Concat', inputs=shape_i, axis=0)])
            out.append(
                graph.make_node(
                    'Expand',
                    inputs=[t_reshaped, out_shape],
                    outputs=node.output('Out')[i]))
