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

from paddle2onnx.constant import dtypes


def slice_helper(graph, input, axes, starts, ends, outputs=None):
    if graph.opset_version < 10:
        slice_node = graph.make_node(
            "Slice",
            inputs=[input],
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


def constant_helper(graph, dtype, value, shape=None, outputs=None):
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
