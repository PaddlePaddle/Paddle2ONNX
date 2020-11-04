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


@op_mapper(
    ['relu', 'tanh', 'log', 'sigmoid'],
    mapper_dict={
        'relu': 'Relu',
        'tanh': 'Tanh',
        'log': 'Log',
        'sigmoid': 'Sigmoid',
    })
class ActivationOps():
    support_opset_verison_range = (1, 12)

    @classmethod
    def opset_1(cls, graph, node, **kw):
        onnx_type = kw['mapper_dict'][node.type]
        onnx_node = graph.make_node(
            onnx_type, inputs=node.input('X'), outputs=node.output('Out'))


@op_mapper('leaky_relu')
class LeakyRelu():
    support_opset_verison_range = (1, 12)

    @classmethod
    def opset_1(cls, graph, node, **kw):
        onnx_node = graph.make_node(
            'LeakyRelu',
            inputs=[node.input('X')[0]],
            outputs=node.output('Out'),
            alpha=node.attr('alpha'))


@op_mapper('prelu')
class PRelu():
    support_opset_verison_range = (1, 12)

    @classmethod
    def opset_1(cls, graph, node, **kw):
        onnx_node = graph.make_node(
            'PRelu',
            inputs=[node.input('X')[0], node.input('Alpha')[0]],
            outputs=node.output('Out'))


@op_mapper('relu6')
class Relu6():
    support_opset_verison_range = (1, 12)

    @classmethod
    def opset_1(cls, graph, node, **kw):
        threshold = node.attr('threshold')
        onnx_node = graph.make_node(
            'Clip',
            inputs=[node.input('X')[0]],
            outputs=node.output('Out'),
            max=threshold,
            min=0.0)

    @classmethod
    def opset_11(cls, graph, node, **kw):
        min_node = graph.make_node(
            'Constant', attrs={'dtype': dtypes.ONNX.FLOAT,
                               'value': 0})
        max_node = graph.make_node(
            'Constant',
            attrs={
                'dtype': dtypes.ONNX.FLOAT,
                'value': node.attr('threshold')
            })
        graph.make_node(
            'Clip',
            inputs=[node.input('X')[0], min_node, max_node],
            outputs=node.output('Out'), )


@op_mapper('hard_sigmoid')
class HardSigmoid():
    support_opset_verison_range = (1, 12)

    @classmethod
    def opset_1(cls, graph, node, **kw):
        slope = node.attr('slope')
        offset = node.attr('offset')
        graph.make_node(
            'HardSigmoid',
            inputs=node.input('X'),
            outputs=node.output('Out'),
            alpha=slope,
            beta=offset)


@op_mapper('swish')
class Swish():
    support_opset_verision_range = (7, 12)

    @classmethod
    def opset_7(cls, graph, node, **kw):
        beta_node = graph.make_node(
            'Constant',
            attrs={'dtype': dtypes.ONNX.FLOAT,
                   'value': [node.attr('beta')]})
        beta_x_node = graph.make_node(
            'Mul', inputs=[node.input('X')[0], beta_node])
        sigmoid_node = graph.make_node('Sigmoid', inputs=[beta_x_node])
        graph.make_node(
            'Mul',
            inputs=[node.input('X')[0], sigmoid_node],
            outputs=node.output('Out'))


@op_mapper('hard_swish')
class HardSwish():
    support_opset_verision_range = (7, 12)

    @classmethod
    def opset_7(cls, graph, node, **kw):
        scale_node = graph.make_node(
            'Constant',
            attrs={'dtype': dtypes.ONNX.FLOAT,
                   'value': node.attr('scale')})
        offset_node = graph.make_node(
            'Constant',
            attrs={'dtype': dtypes.ONNX.FLOAT,
                   'value': node.attr('offset')})

        node0 = graph.make_node('Add', inputs=[node.input('X')[0], offset_node])
        min_value = 0.0
        max_value = node.attr('threshold')
        node1 = graph.make_node(
            'Clip', inputs=[node0], max=max_value, min=min_value)
        node2 = graph.make_node('Mul', inputs=[node.input('X')[0], node1])
        node3 = graph.make_node(
            'Div', inputs=[node2, scale_node], outputs=node.output('Out'))
