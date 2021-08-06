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


@op_mapper('where_index')
class WhereIndex():
    support_opset_verison_range = (9, 13)

    @classmethod
    def opset_9(cls, graph, node, **kw):
        nonzero_node = graph.make_node(
            'NonZero', inputs=node.input('Condition'))
        graph.make_node(
            'Transpose',
            inputs=[nonzero_node],
            outputs=node.output('Out'),
            perm=[1, 0])


@op_mapper('top_k_v2')
class TopKV2():
    support_opset_verison_range = (11, )

    @classmethod
    def opset_11(cls, graph, node, **kw):
        if 'K' in node.inputs and len(node.input('K')) > 0:
            k_node = node.input('K', 0)
            k_node_dtype = node.input_dtype('K', 0)
            if dtypes.DTYPE_PADDLE_STR_MAP[k_node_dtype] != 'int64':
                k_node = graph.make_node(
                    'Cast', inputs=[k_node], to=dtypes.ONNX.INT64)
            graph.make_node(
                'TopK',
                inputs=[node.input('X', 0), k_node],
                outputs=[node.output('Out', 0), node.output('Indices', 0)],
                largest=node.attr('largest'),
                sorted=node.attr('sorted'),
                axis=node.attr('axis'))
        else:
            k = node.attr('k')
            k_node = graph.make_node(
                'Constant', attrs={'dtype': dtypes.ONNX.INT64,
                                   'value': [k]})
            graph.make_node(
                'TopK',
                inputs=[node.input('X', 0), k_node],
                outputs=[node.output('Out', 0), node.output('Indices', 0)],
                largest=node.attr('largest'),
                sorted=node.attr('sorted'),
                axis=node.attr('axis'))

@op_mapper('top_k')
class TopK():
    support_opset_verison_range = (11, )

    @classmethod
    def opset_11(cls, graph, node, **kw):
        if 'K' in node.inputs and len(node.input('K')) > 0:
            k_node = node.input('K', 0)
            k_node_dtype = node.input_dtype('K', 0)
            if dtypes.DTYPE_PADDLE_STR_MAP[k_node_dtype] != 'int64':
                k_node = graph.make_node(
                    'Cast', inputs=[k_node], to=dtypes.ONNX.INT64)
            graph.make_node(
                'TopK',
                inputs=[node.input('X', 0), k_node],
                outputs=[node.output('Out', 0), node.output('Indices', 0)])
        else:
            k = node.attr('k')
            k_node = graph.make_node(
                'Constant', attrs={'dtype': dtypes.ONNX.INT64,
                                   'value': [k]})
            graph.make_node(
                'TopK',
                inputs=[node.input('X', 0), k_node],
                outputs=[node.output('Out', 0), node.output('Indices', 0)])
