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

import paddle.fluid.core as core
import six
import copy
from paddle2onnx.constant import dtypes
import paddle


def is_static_shape(shape):
    if len(shape) > 1 and shape[1:].count(-1) > 0:
        raise Exception(
            "Converting this model to ONNX need with static input shape," \
            " please fix input shape of this model, see doc Q2 in" \
            " https://github.com/PaddlePaddle/paddle2onnx/blob/develop/docs/en/FAQ.md."
        )


def slice_helper(graph, input, axes, starts, ends, outputs=[]):
    if graph.opset_version < 10:
        slice_node = graph.make_node(
            "Slice",
            inputs=input,
            outputs=outputs,
            axes=axes,
            starts=starts,
            ends=ends)
        return slice_node
    else:
        axes_node = graph.make_node(
            'Constant', dtype=dtypes.ONNX.INT64, value=axes)
        starts_node = graph.make_node(
            'Constant', dtype=dtypes.ONNX.INT64, value=starts)
        ends_node = graph.make_node(
            'Constant', dtype=dtypes.ONNX.INT64, value=ends)
        slice_node = graph.make_node(
            "Slice",
            inputs=[input, starts_node, ends_node, axes_node],
            outputs=outputs)
        return slice_node


def squeeze_helper(graph, input, axes):
    if graph.opset_version < 13:
        squeeze_node = graph.make_node("Squeeze", inputs=input, axes=axes)
        return squeeze_node
    else:
        axes_node = graph.make_node(
            'Constant', dtype=dtypes.ONNX.INT64, value=axes)
        squeeze_node = graph.make_node("Squeeze", inputs=[input, axes_node])
        return squeeze_node


def constant_helper(graph, dtype, value, shape=None, outputs=[]):
    constant = graph.make_node(
        'Constant',
        inputs=[],
        outputs=outputs,
        attrs={
            'dims': shape,
            'dtype': dtypes.DTYPE_PADDLE_ONNX_MAP[dtype],
            'value': value
        })
    return constant


def clip_helper(graph, node, input, max, min, output=[]):
    x_dtype = node.input_dtype('X', 0)
    if (isinstance(min, six.string_types) or
            isinstance(max, six.string_types)) and graph.opset_version < 11:
        raise Exception(
            "min or max of Clip is Tensor, please try with higher onnx opset_version."
        )
    if graph.opset_version < 11:
        if x_dtype != paddle.float32:
            input = graph.make_node(
                'Cast', inputs=[input], to=dtypes.ONNX.FLOAT)
            clip = graph.make_node('Clip', inputs=input, max=max, min=min)
            clip = graph.make_node(
                'Cast',
                inputs=[clip],
                to=dtypes.DTYPE_PADDLE_ONNX_MAP[x_dtype],
                outputs=output)
        else:
            clip = graph.make_node(
                'Clip', inputs=input, max=max, min=min, outputs=output)
    else:
        if not isinstance(min, six.string_types):
            min = graph.make_node(
                'Constant',
                attrs={
                    'dtype': dtypes.DTYPE_PADDLE_ONNX_MAP[x_dtype],
                    'value': min
                })
        else:
            if node.input_dtype('Min', 0) != x_dtype:
                min = graph.make_node(
                    'Cast',
                    inputs=min,
                    attrs={'to': dtypes.DTYPE_PADDLE_ONNX_MAP[x_dtype]})
            min = graph.make_node('Squeeze', min)

        if not isinstance(max, six.string_types):
            max = graph.make_node(
                'Constant',
                attrs={
                    'dtype': dtypes.DTYPE_PADDLE_ONNX_MAP[x_dtype],
                    'value': max
                })
        else:
            if node.input_dtype('Max', 0) != x_dtype:
                max = graph.make_node(
                    'Cast',
                    inputs=max,
                    attrs={'to': dtypes.DTYPE_PADDLE_ONNX_MAP[x_dtype]})
            max = graph.make_node('Squeeze', max)

        clip = graph.make_node('Clip', inputs=[input, min, max], outputs=output)
    return clip


def dtype_alignment(graph, nodes, node_dtypes):
    assert len(nodes) == len(
        node_dtypes), "Length of nodes and node_dtypes should be equal."
    dtype_order = [
        core.VarDesc.VarType.BOOL,
        core.VarDesc.VarType.INT16,
        core.VarDesc.VarType.INT32,
        core.VarDesc.VarType.INT64,
        core.VarDesc.VarType.FP16,
        core.VarDesc.VarType.FP32,
        core.VarDesc.VarType.FP64,
    ]
    max_index = -1
    for dtype in node_dtypes:
        index = dtype_order.index(dtype)
        if index > max_index:
            max_index = index

    if max_index < 0:
        return nodes

    casted_nodes = list()
    cast_dtype = dtype_order[max_index]
    cast_dtype = dtypes.DTYPE_PADDLE_ONNX_MAP[cast_dtype]
    for i, dtype in enumerate(node_dtypes):
        index = dtype_order.index(dtype)
        if index != max_index:
            cast_node = graph.make_node(
                'Cast', inputs=[nodes[i]], to=cast_dtype)
            casted_nodes.append(cast_node)
        else:
            casted_nodes.append(nodes[i])
    return casted_nodes


def cast(graph, input, origin_dtype, target_dtype):
    if not isinstance(origin_dtype, six.string_types):
        origin_dtype = dtypes.DTYPE_PADDLE_STR_MAP[origin_dtype]
    if origin_dtype != target_dtype:
        cast_node = graph.make_node(
            'Cast', inputs=input, to=dtypes.DTYPE_ONNX_STR_MAP[target_dtype])
        return cast_node
    return input


def shape_alignment(graph, nodes, node_shapes):
    assert len(nodes) == len(
        node_shapes), "Length of nodes and node_shapes should be equal."
    max_dim = -1
    for shape in node_shapes:
        dim = len(shape)
        if dim > max_dim:
            max_dim = dim

    if max_dim < 0:
        return nodes

    assert max_dim == 1 or max_dim == 0, "max_dim is only supported when max_dim is 1 or 0."
    max_dim = 1 if max_dim == 0 else max_dim
    unsqueeze_nodes = list()
    for i, shape in enumerate(node_shapes):
        dim = len(shape)
        if dim != max_dim:
            unsqueeze_node = nodes[i]
            for j in range(max_dim - dim):
                if graph.opset_version < 13:
                    unsqueeze_node = graph.make_node(
                        'Unsqueeze', inputs=[unsqueeze_node], axes=[0])
                else:
                    axes_node = graph.make_node(
                        'Constant',
                        attrs={'dtype': dtypes.ONNX.INT64,
                               'value': 0})
                    unsqueeze_node = graph.make_node(
                        'Unsqueeze', inputs=[unsqueeze_node, axes_node])
            unsqueeze_nodes.append(unsqueeze_node)
        else:
            unsqueeze_nodes.append(nodes[i])
    return unsqueeze_nodes


def get_tensor_list_node(graph, node, name):
    node_list = node.input(name)
    node_dtypes = [node.input_dtype(name, i) for i in range(len(node_list))]
    node_list = dtype_alignment(graph, node_list, node_dtypes)

    node_shapes = [node.input_shape(name, i) for i in range(len(node_list))]
    node_list = shape_alignment(graph, node_list, node_shapes)
    node = graph.make_node("Concat", inputs=node_list, axis=0)
    return node


def get_value_from_parameters(graph, input_node):
    if input_node not in graph.parameters:
        raise Exception(
            "Currently does not support the input node parameter as input tensor!,"
            "Try converting with opset_version > { } ".format(
                graph.opset_version))
    else:
        data = graph.parameters[input_node].attribute[0].t.int32_data
        if data is None or len(data) < 1:
            data = graph.parameters[input_node].attribute[0].t.int64_data
        value = [val for _, val in enumerate(data)]
    return value


def func(graph, axes_node):
    if axes_node not in graph.parameters:
        raise Exception(
            "Currently does not support the axis parameter as input tensor!")
    else:
        axes = graph.parameters[axes_node].attribute[0].t.int32_data
        if axes is None or len(axes) < 1:
            axes = graph.parameters[axes_node].attribute[0].t.int64_data
        return axes
