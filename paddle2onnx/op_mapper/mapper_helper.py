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


def clip_helper(graph, input, max, min, output=[]):
    if (isinstance(min, six.string_types) or
            isinstance(max, six.string_types)) and graph.opset_version < 11:
        raise "min or max of Clip is Tensor, please try with higher onnx opset_version."
    if graph.opset_version < 11:
        clip = graph.make_node(
            'Clip', inputs=input, max=max, min=min, outputs=output)
    else:
        if not isinstance(min, six.string_types):
            min = graph.make_node(
                'Constant', attrs={'dtype': dtypes.ONNX.FLOAT,
                                   'value': min})
        else:
            min = graph.make_node('Squeeze', min, axes=[0])
        if not isinstance(max, six.string_types):
            max = graph.make_node(
                'Constant', attrs={'dtype': dtypes.ONNX.FLOAT,
                                   'value': max})
        else:
            max = graph.make_node('Squeeze', max, axes=[0])
        clip = graph.make_node('Clip', inputs=[input, min, max], outputs=output)
    return clip


def dtype_alignment(graph, nodes, node_dtypes):
    assert len(nodes) == len(
        node_dtypes), "Length of nodes and node_dtypes should be equal."
    dtype_order = [
        core.VarDesc.VarType.BOOL, core.VarDesc.VarType.INT16,
        core.VarDesc.VarType.INT32, core.VarDesc.VarType.INT64,
        core.VarDesc.VarType.FP16, core.VarDesc.VarType.FP32
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
