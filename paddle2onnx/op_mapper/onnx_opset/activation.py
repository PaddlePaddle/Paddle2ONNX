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

import math
import sys
import os
import numpy as np
import paddle.fluid.core as core
import paddle.fluid as fluid
import onnx
from onnx import onnx_pb
from paddle2onnx.constant import dtypes
from paddle2onnx.op_mapper import OpMapper as op_mapper


@op_mapper(
    ['relu', 'tanh', 'log', 'sigmoid', 'leaky_relu'],
    mapper_dict={
        'relu': 'Relu',
        'tanh': 'Tanh',
        'log': 'Log',
        'sigmoid': 'Sigmoid',
    })
class ActivationOps():
    @classmethod
    def opset_9(cls, graph, node, **kw):
        onnx_type = kw['mapper_dict'][node.type]
        onnx_node = graph.update_node(
            node, onnx_type, inputs=node.input('X'), outputs=node.output('Out'))


@op_mapper('leaky_relu')
class LeakyRelu():
    @classmethod
    def opset_9(cls, graph, node, **kw):
        onnx_node = graph.update_node(
            node,
            'LeakyRelu',
            inputs=[node.input('X')[0]],
            outputs=node.output('Out'),
            alpha=node.attr('alpha'))


@op_mapper('prelu')
class PRelu():
    @classmethod
    def opset_9(cls, graph, node, **kw):
        onnx_node = graph.update_node(
            node,
            'PRelu',
            inputs=[node.input('X')[0], node.input('Alpha')[0]],
            outputs=node.output('Out'))
        return onnx_node


@op_mapper('relu6')
class Relu6():
    @classmethod
    def opset_9(cls, graph, node, **kw):
        threshold = node.attr('threshold')
        onnx_node = graph.update_node(
            node,
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
        graph.update_node(
            node,
            'Clip',
            inputs=[node.input('X')[0], min_node, max_node],
            outputs=node.output('Out'), )


@op_mapper('hard_sigmoid')
class HardSigmoid():
    @classmethod
    def opset_9(cls, graph, node, **kw):
        slope = node.attr('slope')
        offset = node.attr('offset')
        graph.update_node(
            node,
            'HardSigmoid',
            inputs=node.input('X'),
            outputs=node.output('Out'),
            alpha=slope,
            beta=offset)


@op_mapper('swish')
class Swish():
    @classmethod
    def opset_9(cls, graph, node, **kw):
        beta_node = graph.make_node(
            'Constant',
            attrs={'dtype': dtypes.ONNX.FLOAT,
                   'value': [node.attr('beta')]})
        beta_x_node = graph.make_node(
            'Mul', inputs=[node.input('X')[0], beta_node])
        sigmoid_node = graph.make_node('Sigmoid', inputs=[beta_x_node])
        graph.update_node(
            node,
            'Mul',
            inputs=[node.input('X')[0], sigmoid_node],
            outputs=node.output('Out'))


@op_mapper('hard_swish')
class HardSwish():
    @classmethod
    def opset_9(cls, graph, node, **kw):
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
        graph.remove_node(node)
