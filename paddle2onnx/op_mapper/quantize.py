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
from paddle2onnx.utils import logging
from onnx import helper
import onnx


@op_mapper('quantize_linear')
class QuantizeLinear():
    support_opset_version_range = (13, 15)

    @classmethod
    def opset_13(cls, graph, node, **kw):
        scale_key = node.input('Scale', 0)
        param, weight_scale = mapper_helper.get_param_from_paddle_graph(
            graph, scale_key)
        weight_scale = weight_scale / 127

        zero_node = graph.make_node(
            'Constant', dtype=dtypes.ONNX.INT8, value=[0])

        scale_list = weight_scale.squeeze().tolist()
        scale_node = graph.make_node(
            'Constant', dtype=dtypes.ONNX.FLOAT, value=scale_list)
        graph.quantize_params_dict[node.input('X', 0)] = [
            scale_node, zero_node, scale_list, [0], node.attr("quant_axis")
        ]

        quantize_node = graph.make_node(
            'QuantizeLinear',
            inputs=[node.input('X', 0), scale_node, zero_node],
            outputs=node.output('Y'),
            axis=node.attr("quant_axis"))


@op_mapper('dequantize_linear')
class DequantizeLinear():
    support_opset_version_range = (13, 15)

    @classmethod
    def opset_13(cls, graph, node, **kw):
        scale_key = node.input('Scale', 0)
        param, weight_scale = mapper_helper.get_param_from_paddle_graph(
            graph, scale_key)
        weight_scale = weight_scale / 127

        scale_list = weight_scale.squeeze().tolist()
        if not isinstance(scale_list, list):
            scale_list = [scale_list]
        scale_node = graph.make_node(
            'Constant', dtype=dtypes.ONNX.FLOAT, value=scale_list)
        zero_node = graph.make_node(
            'Constant', dtype=dtypes.ONNX.INT8, value=[0] * len(scale_list))

        key = node.input('X', 0)
        update_param, weight = mapper_helper.get_param_from_paddle_graph(graph,
                                                                         key)
        if node.input('X', 0) not in graph.quantize_params_dict:
            graph.quantize_params_dict[node.input('X', 0)] = [
                scale_node, zero_node, scale_list, [0] * len(scale_list),
                node.attr("quant_axis")
            ]
        quantize_node = node.input('X', 0)
        if weight is not None:
            if len(weight.shape) == 4:
                new_weight = weight.transpose(1, 2, 3, 0) * weight_scale
                new_weight = new_weight.transpose(3, 0, 1, 2)
            else:
                new_weight = weight * weight_scale
            update_param["data"] = new_weight
            graph.update_parameters(key, update_param)

            quantize_node = graph.make_node(
                'QuantizeLinear',
                inputs=[node.input('X', 0), scale_node, zero_node],
                axis=node.attr("quant_axis"))

        dequantize_node = graph.make_node(
            'DequantizeLinear',
            inputs=[quantize_node, scale_node, zero_node],
            outputs=[node.output('Y', 0)],
            axis=node.attr("quant_axis"))


@op_mapper('fake_quantize_dequantize_moving_average_abs_max')
class FakeQuantizeDequantizeMovingAverageAbsMax():
    support_opset_version_range = (10, 13)

    @classmethod
    def opset_13(cls, graph, node, **kw):
        zero_node = graph.make_node('Constant', dtype=dtypes.ONNX.INT8, value=0)

        key = node.input('InScale', 0)
        _, in_scale = mapper_helper.get_param_from_paddle_graph(graph, key)

        input_node_name = node.input('X', 0)

        scale_list = float(in_scale[0]) / 127.0
        scale_node = graph.make_node(
            'Constant', dtype=dtypes.ONNX.FLOAT, value=scale_list)

        quantize_node = graph.make_node(
            'QuantizeLinear', inputs=[input_node_name, scale_node, zero_node])
        graph.make_node(
            'DequantizeLinear',
            inputs=[quantize_node, scale_node, zero_node],
            outputs=node.output('Out', 0))

        graph.quantize_params_dict[node.input('X', 0)] = [
            scale_node, zero_node, [scale_list], [0], node.attr("quant_axis")
        ]


@op_mapper('fake_channel_wise_quantize_dequantize_abs_max')
class FakeChannelWiseQuantizeDequantizeAbsMax():
    support_opset_version_range = (10, 13)

    @classmethod
    def opset_13(cls, graph, node, **kw):
        quant_axis = node.attr("quant_axis")
        key = node.input('X', 0)
        param, weight = mapper_helper.get_param_from_paddle_graph(graph, key)

        weight_numpy = np.array(weight)
        scale_list, zero_list = mapper_helper.quantize_weight_per_channel(
            weight_numpy, quant_axis)
        scale = np.array(scale_list)
        zero_node = graph.make_node(
            'Constant', dtype=dtypes.ONNX.INT8, value=zero_list)
        scale_node = graph.make_node(
            'Constant', dtype=dtypes.ONNX.FLOAT, value=scale_list)
        graph.quantize_params_dict[node.input('X', 0)] = [
            scale_node, zero_node, scale_list, zero_list, quant_axis
        ]
        quantize_node = graph.make_node(
            'QuantizeLinear',
            inputs=[node.input('X', 0), scale_node, zero_node],
            axis=quant_axis)

        graph.make_node(
            'DequantizeLinear',
            inputs=[quantize_node, scale_node, zero_node],
            outputs=node.output('Out'),
            axis=quant_axis)


@op_mapper('moving_average_abs_max_scale')
class MovingAverageAbsMaxScale():
    support_opset_version_range = (10, 13)

    @classmethod
    def opset_13(cls, graph, node, **kw):
        graph.make_node(
            'Identity', inputs=node.input('X'), outputs=node.output('Out'))


@op_mapper('fake_quantize_dequantize_abs_max')
class FakeQuantizeDequantizeAbsMax():
    support_opset_version_range = (10, 13)

    @classmethod
    def opset_10(cls, graph, node, **kw):
        key = node.input('X', 0)
        update_param, weight = mapper_helper.get_param_from_paddle_graph(graph,
                                                                         key)
        weight_numpy = np.array(weight)
        scale_list, zero_list = mapper_helper.quantize_weight(weight_numpy)

        scale_node = graph.make_node(
            'Constant', dtype=dtypes.ONNX.FLOAT, value=scale_list[0])
        zero_node = graph.make_node('Constant', dtype=dtypes.ONNX.INT8, value=0)
        graph.quantize_params_dict[node.input(
            'X', 0)] = [scale_node, zero_node, scale_list, [0], -1]
        quantize_node = graph.make_node(
            'QuantizeLinear',
            inputs=[node.input('X', 0), scale_node, zero_node])

        graph.make_node(
            'DequantizeLinear',
            inputs=[quantize_node, scale_node, zero_node],
            outputs=node.output('Out'))


@op_mapper('fake_quantize_range_abs_max')
class FakeQuantizeRangeAbsMax():
    support_opset_version_range = (10, 13)

    @classmethod
    def opset_10(cls, graph, node, **kw):
        zero_node = graph.make_node('Constant', dtype=dtypes.ONNX.INT8, value=0)

        key = node.input('InScale', 0)
        _, in_scale = mapper_helper.get_param_from_paddle_graph(graph, key)

        input_node_name = node.input('X', 0)

        scale_list = [float(in_scale[0]) / 127.0]
        scale_node = graph.make_node(
            'Constant', dtype=dtypes.ONNX.FLOAT, value=scale_list[0])

        assert graph.quantize_model_mode in [
            "static"
        ], "fake_quantize_range_abs_max only can be in static quantize model"
        quantize_node = graph.make_node(
            'QuantizeLinear', inputs=[input_node_name, scale_node, zero_node])
        graph.make_node(
            'DequantizeLinear',
            inputs=[quantize_node, scale_node, zero_node],
            outputs=node.output('Out', 0))

        graph.quantize_params_dict[node.input(
            'X', 0)] = [scale_node, zero_node, scale_list, [0], -1]


@op_mapper('fake_quantize_moving_average_abs_max')
class FakeQuantizeMovingAverageAbsMax():
    support_opset_version_range = (10, 13)

    @classmethod
    def opset_13(cls, graph, node, **kw):
        assert graph.quantize_model_mode in [
            "static"
        ], "fake_quantize_moving_average_abs_max only can be in static quantize model"
        zero_node = graph.make_node('Constant', dtype=dtypes.ONNX.INT8, value=0)

        key = node.input('InScale', 0)
        _, in_scale = mapper_helper.get_param_from_paddle_graph(graph, key)

        input_node_name = node.input('X', 0)

        scale_list = [float(in_scale[0]) / 127.0]
        scale_node = graph.make_node(
            'Constant', dtype=dtypes.ONNX.FLOAT, value=scale_list[0])
        quantize_node = graph.make_node(
            'QuantizeLinear', inputs=[input_node_name, scale_node, zero_node])
        graph.make_node(
            'DequantizeLinear',
            inputs=[quantize_node, scale_node, zero_node],
            outputs=node.output('Out', 0))
        graph.quantize_params_dict[node.input(
            'X', 0)] = [scale_node, zero_node, scale_list, [0], -1]


@op_mapper('fake_dequantize_max_abs')
class FakeDequantizeMaxAbs():
    support_opset_version_range = (10, 13)

    @classmethod
    def opset_13(cls, graph, node, **kw):
        assert graph.quantize_model_mode in [
            "static"
        ], "fake_dequantize_max_abs only can be in static quantize model"


@op_mapper('fake_channel_wise_quantize_abs_max')
class FakeChannelWiseQuantizeAbsMax():
    support_opset_version_range = (10, 13)

    @classmethod
    def opset_13(cls, graph, node, **kw):
        assert graph.quantize_model_mode in [
            "static"
        ], "fake_channel_wise_quantize_abs_max only can be in static quantize model"
        quant_axis = node.attr("quant_axis")
        key = node.output('OutScale', 0)
        _, scale = mapper_helper.get_param_from_paddle_graph(graph, key)
        scale_list = np.squeeze(scale).tolist()
        zero_list = [0] * len(scale_list)
        zero_node = graph.make_node(
            'Constant', dtype=dtypes.ONNX.INT8, value=zero_list)

        scale_node = graph.make_node(
            'Constant', dtype=dtypes.ONNX.FLOAT, value=scale_list)

        quantize_node = graph.make_node(
            'QuantizeLinear',
            inputs=[node.input('X', 0), scale_node, zero_node],
            axis=quant_axis)
        graph.make_node(
            'DequantizeLinear',
            inputs=[quantize_node, scale_node, zero_node],
            outputs=node.output('Out'),
            axis=quant_axis)
        graph.quantize_params_dict[node.input('X', 0)] = [
            scale_node, zero_node, scale_list, zero_list, quant_axis
        ]


@op_mapper('fake_channel_wise_dequantize_max_abs')
class FakeChannelWiseDequantizeMaxAbs():
    support_opset_version_range = (10, 13)

    @classmethod
    def opset_13(cls, graph, node, **kw):
        assert graph.quantize_model_mode in [
            "static"
        ], "fake_channel_wise_dequantize_max_abs only can be in static quantize model"
