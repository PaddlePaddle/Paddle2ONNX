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
import paddle


@op_mapper(
    ['relu', 'tanh', 'log', 'sigmoid', 'sqrt'],
    mapper_dict={
        'relu': 'Relu',
        'tanh': 'Tanh',
        'log': 'Log',
        'sigmoid': 'Sigmoid',
        'sqrt': 'Sqrt',
    })
class ActivationOps():
    support_opset_version_range = (7, 15)

    @classmethod
    def opset_1(cls, graph, node, **kw):
        onnx_type = kw['mapper_dict'][node.type]
        onnx_node = graph.make_node(
            onnx_type, inputs=node.input('X'), outputs=node.output('Out'))


@op_mapper('leaky_relu')
class LeakyRelu():
    support_opset_version_range = (7, 15)

    @classmethod
    def opset_1(cls, graph, node, **kw):
        onnx_node = graph.make_node(
            'LeakyRelu',
            inputs=[node.input('X')[0]],
            outputs=node.output('Out'),
            alpha=node.attr('alpha'))


@op_mapper('softplus')
class Softplus():
    support_opset_version_range = (7, 15)

    @classmethod
    def opset_1(cls, graph, node, **kw):
        beta = node.attr('beta')
        threshold = node.attr('threshold')
        if np.isclose(beta, 1.0, 1e-06, 1e-06) and \
            np.isclose(threshold, 20.0, 1e-06, 1e-06):
            onnx_node = graph.make_node(
                'Softplus',
                inputs=[node.input('X')[0]],
                outputs=node.output('Out'))
        else:
            raise Exception("[ERROR] Operator softplus " \
            "only supported while beta==1.0 and threshold==20.0")


@op_mapper('prelu')
class PRelu():
    support_opset_version_range = (7, 15)

    @classmethod
    def opset_7(cls, graph, node, **kw):
        x = node.input('X')[0]
        x_dtype = node.input_dtype('X', 0)
        if x_dtype == paddle.float64:
            x = graph.make_node('Cast', inputs=[x], to=dtypes.ONNX.FLOAT)

        slope_node = node.input('Alpha')[0]
        slope_dtype = node.input_dtype('Alpha', 0)
        if slope_dtype == paddle.float64:
            slope_node = graph.make_node(
                'Cast', inputs=[slope_node], to=dtypes.ONNX.FLOAT)
        if x_dtype == paddle.float64:
            prelu_node = graph.make_node('PRelu', inputs=[x, slope_node])
            graph.make_node(
                'Cast',
                inputs=[prelu_node],
                to=dtypes.ONNX.DOUBLE,
                outputs=node.output('Out'))
        else:
            prelu_node = graph.make_node(
                'PRelu', inputs=[x, slope_node], outputs=node.output('Out'))


@op_mapper('relu6')
class Relu6():
    support_opset_version_range = (7, 15)

    @classmethod
    def opset_1(cls, graph, node, **kw):
        mapper_helper.clip_helper(graph, node,
                                  node.input('X', 0),
                                  node.attr('threshold'), 0.0,
                                  node.output('Out', 0))


@op_mapper('gelu')
class Gelu():
    support_opset_version_range = (9, 15)

    @classmethod
    def opset_9(cls, graph, node, **kw):
        input = node.input('X', 0)
        x_dtype = node.input_dtype('X', 0)
        # onnxruntime only support float32 Erf
        if x_dtype != paddle.float32:
            input = graph.make_node(
                'Cast', inputs=[input], to=dtypes.ONNX.FLOAT)
        sqrt2 = graph.make_node(
            'Constant', dtype=dtypes.ONNX.FLOAT, value=[1.4142135623730951])
        zero_point_five = graph.make_node(
            'Constant', dtype=dtypes.ONNX.FLOAT, value=[0.5])
        one = graph.make_node('Constant', dtype=dtypes.ONNX.FLOAT, value=[1])
        x = graph.make_node('Div', inputs=[input, sqrt2])
        x = graph.make_node('Erf', inputs=x)
        x = graph.make_node('Add', inputs=[x, one])
        x = graph.make_node('Mul', inputs=[input, x])
        if x_dtype != paddle.float32:
            mul_node = graph.make_node('Mul', inputs=[x, zero_point_five])
            graph.make_node(
                'Cast',
                inputs=[mul_node],
                to=dtypes.DTYPE_PADDLE_ONNX_MAP[x_dtype],
                outputs=node.output('Out'))
        else:
            graph.make_node(
                'Mul', inputs=[x, zero_point_five], outputs=node.output('Out'))


@op_mapper('selu')
class Selu():
    support_opset_version_range = (7, 15)

    @classmethod
    def opset_6(cls, graph, node, **kw):
        graph.make_node(
            'Selu',
            inputs=node.input('X'),
            alpha=node.attr('alpha'),
            gamma=node.attr('scale'),
            outputs=node.output('Out'))


@op_mapper('hard_sigmoid')
class HardSigmoid():
    support_opset_version_range = (7, 15)

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
    support_opset_version_range = (7, 15)

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
    support_opset_version_range = (7, 15)

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
        node1 = mapper_helper.clip_helper(graph, node, node0,
                                          node.attr('threshold'), 0.0)
        node2 = graph.make_node('Mul', inputs=[node.input('X')[0], node1])
        node3 = graph.make_node(
            'Div', inputs=[node2, scale_node], outputs=node.output('Out'))
