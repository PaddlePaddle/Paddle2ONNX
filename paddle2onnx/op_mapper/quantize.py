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
import paddle
from onnx import helper
import onnx


@op_mapper('fake_quantize_dequantize_moving_average_abs_max')
class Fake_quantize_dequantize_moving_average_abs_max():
    support_opset_version_range = (10, 13)

    @classmethod
    def opset_13(cls, graph, node, **kw):
        zero_node = graph.make_node(
            'Constant', dtype=dtypes.ONNX.INT8, value=[0])

        key = node.input('InScale', 0)
        InScale = None
        if key in graph.origin_parameters.keys():
            param = graph.origin_parameters[key]
            InScale = param['data']

        Minus_InScale = -1.0 * InScale

        input_node_name = node.input('X', 0)

        scale_node = graph.make_node(
            'Constant',
            dtype=dtypes.ONNX.FLOAT,
            value=[float(InScale[0]) / 127.0])

        graph.add_name(node.output('Out', 0))
        output_name = graph.get_name(node.output('Out', 0))

        quantize_node = graph.make_node(
            'QuantizeLinear', inputs=[input_node_name, scale_node, zero_node])
        graph.make_node(
            'DequantizeLinear',
            inputs=[quantize_node, scale_node, zero_node],
            outputs=output_name)

        if input_node_name in graph.changed_dict.keys():
            return

        another_nodes = graph.get_another_node_by_input(
            input_node_name, copy_node=False)
        if len(another_nodes) == 0:
            return
        index = 0
        all_q_dq = list()
        for another_node in another_nodes:
            zero_node, z_node = graph.make_node(
                'Constant', dtype=dtypes.ONNX.INT8, value=[0], return_node=True)
            scale_node, s_node = graph.make_node(
                'Constant',
                dtype=dtypes.ONNX.FLOAT,
                value=[float(InScale[0]) / 127.0],
                return_node=True)
            changed_output_name = output_name + ".paddleadd" + str(index)
            index = index + 1
            quantize_node, q_node = graph.make_node(
                'QuantizeLinear',
                inputs=[input_node_name, scale_node, zero_node],
                return_node=True)
            dequantize_node, dq_node = graph.make_node(
                'DequantizeLinear',
                inputs=[quantize_node, scale_node, zero_node],
                outputs=changed_output_name,
                return_node=True)
            all_q_dq.append([
                z_node.layer_name, s_node.layer_name, q_node.layer_name,
                dq_node.layer_name
            ])

        graph.changed_dict[input_node_name] = dict()
        graph.changed_dict[input_node_name]["name"] = output_name
        graph.changed_dict[input_node_name]["total"] = index
        graph.changed_dict[input_node_name]["qdq"] = all_q_dq


@op_mapper('fake_channel_wise_quantize_dequantize_abs_max')
class Fake_channel_wise_quantize_dequantize_abs_max():
    support_opset_version_range = (10, 13)

    @classmethod
    def opset_13(cls, graph, node, **kw):
        paddle.disable_static()

        key = node.input('X', 0)
        update_param = None
        update_name = None
        weight = None
        if key in graph.origin_parameters.keys():
            param = graph.origin_parameters[key]
            weight = param['data']
            update_name = key
            update_param = param

        tensor = paddle.to_tensor(weight)
        abs_tensor = paddle.abs(tensor)

        val = None
        scale = None
        input_shape = node.input_shape('X', 0)
        zero_node = None
        if len(input_shape) == 4:
            reshape_tensor = paddle.reshape(
                abs_tensor, shape=[input_shape[0], -1])
            vals, _ = paddle.topk(reshape_tensor, k=1, axis=1)
            scale = vals / 127.0
            scale = paddle.unsqueeze(scale, axis=[2, 3])
            zero_node = graph.make_node(
                'Constant', dtype=dtypes.ONNX.INT8, value=[0] * input_shape[0])

        if len(input_shape) == 2:
            vals, _ = paddle.topk(abs_tensor, k=1, axis=0)
            scale = vals / 127.0
            zero_node = graph.make_node(
                'Constant', dtype=dtypes.ONNX.INT8, value=[0] * input_shape[1])

        div = paddle.divide(tensor, scale)

        numpy_data = np.around(div.numpy())
        round_tensor = paddle.to_tensor(numpy_data)

        round_numpy = np.round(div.numpy())

        clip_tensor = round_tensor
        dq_tensor = paddle.multiply(clip_tensor, scale)

        update_param['data'] = dq_tensor.numpy()
        graph.update_parameters(update_name, update_param)

        scale_numpy = paddle.squeeze(scale).numpy().tolist()

        scale_node = graph.make_node(
            'Constant', dtype=dtypes.ONNX.FLOAT, value=scale_numpy)

        attrs = {'axis': node.attr("quant_axis"), }
        quantize_node = graph.make_node(
            'QuantizeLinear',
            inputs=[node.input('X', 0), scale_node, zero_node],
            attrs=attrs)

        graph.make_node(
            'DequantizeLinear',
            inputs=[quantize_node, scale_node, zero_node],
            outputs=node.output('Out'),
            attrs=attrs)


@op_mapper('moving_average_abs_max_scale')
class Moving_average_abs_max_scale():
    support_opset_version_range = (10, 13)

    @classmethod
    def opset_13(cls, graph, node, **kw):
        zero_node = graph.make_node(
            'Constant', dtype=dtypes.ONNX.FLOAT, value=[0.])

        graph.make_node(
            'Add',
            inputs=[node.input('X', 0), zero_node],
            outputs=node.output('Out'))


@op_mapper('fake_quantize_dequantize_abs_max')
class Fake_quantize_dequantize_abs_max():
    support_opset_version_range = (10, 13)

    @classmethod
    def opset_10(cls, graph, node, **kw):
        paddle.disable_static()
        key = node.input('X', 0)
        update_param = None
        update_name = None
        weight = None
        if key in graph.origin_parameters.keys():
            param = graph.origin_parameters[key]
            weight = param['data']
            update_name = key
            update_param = param

        tensor = paddle.to_tensor(weight)
        abs_tensor = paddle.abs(tensor)

        reshape_tensor = paddle.reshape(abs_tensor, shape=[-1])
        vals, _ = paddle.topk(reshape_tensor, k=1)

        scale = vals / 127.0
        zero_node = graph.make_node(
            'Constant', dtype=dtypes.ONNX.INT8, value=[0])

        div = paddle.divide(tensor, scale)

        numpy_data = np.around(div.numpy())
        round_tensor = paddle.to_tensor(numpy_data)

        round_numpy = np.round(div.numpy())

        clip_tensor = round_tensor
        dq_tensor = paddle.multiply(clip_tensor, scale)

        update_param['data'] = dq_tensor.numpy()
        graph.update_parameters(update_name, update_param)

        scale_numpy = paddle.squeeze(scale).numpy().tolist()

        scale_node = graph.make_node(
            'Constant', dtype=dtypes.ONNX.FLOAT, value=scale_numpy)

        quantize_node = graph.make_node(
            'QuantizeLinear',
            inputs=[node.input('X', 0), scale_node, zero_node])

        graph.make_node(
            'DequantizeLinear',
            inputs=[quantize_node, scale_node, zero_node],
            outputs=node.output('Out'))


@op_mapper('fake_quantize_range_abs_max')
class Fake_quantize_range_abs_max():
    support_opset_version_range = (10, 13)

    @classmethod
    def opset_10(cls, graph, node, **kw):
        zero_node = graph.make_node(
            'Constant', dtype=dtypes.ONNX.INT8, value=[0])

        key = node.input('InScale')
        InScale = None
        for name, param in graph.origin_parameters.items():
            if name in key:
                InScale = param['data']

        input_node_name = node.input('X', 0)

        scale_node = graph.make_node(
            'Constant',
            dtype=dtypes.ONNX.FLOAT,
            value=[float(InScale[0]) / 127.0])

        graph.add_name(node.output('Out', 0))
        output_name = graph.get_name(node.output('Out', 0))

        if graph.static_quantize_model:
            quantize_node = graph.make_node(
                'QuantizeLinear',
                inputs=[input_node_name, scale_node, zero_node])
            graph.make_node(
                'DequantizeLinear',
                inputs=[quantize_node, scale_node, zero_node],
                outputs=output_name)
        else:
            quantize_node = graph.make_node(
                'QuantizeLinear',
                inputs=[input_node_name, scale_node, zero_node],
                outputs=output_name)

        if input_node_name in graph.changed_dict.keys():
            return

        another_nodes = graph.get_another_node_by_input(
            input_node_name, copy_node=False)
        if len(another_nodes) == 0:
            return
        index = 0
        all_q_dq = list()
        for another_node in another_nodes:
            zero_node, z_node = graph.make_node(
                'Constant', dtype=dtypes.ONNX.INT8, value=[0], return_node=True)
            scale_node, s_node = graph.make_node(
                'Constant',
                dtype=dtypes.ONNX.FLOAT,
                value=[float(InScale[0]) / 127.0],
                return_node=True)
            changed_output_name = output_name + ".paddleadd" + str(index)
            index = index + 1
            if graph.static_quantize_model:
                quantize_node, q_node = graph.make_node(
                    'QuantizeLinear',
                    inputs=[input_node_name, scale_node, zero_node],
                    return_node=True)
                dequantize_node, dq_node = graph.make_node(
                    'DequantizeLinear',
                    inputs=[quantize_node, scale_node, zero_node],
                    outputs=changed_output_name,
                    return_node=True)
                all_q_dq.append([
                    z_node.layer_name, s_node.layer_name, q_node.layer_name,
                    dq_node.layer_name
                ])
            else:
                quantize_node, q_node = graph.make_node(
                    'QuantizeLinear',
                    inputs=[input_node_name, scale_node, zero_node],
                    outputs=changed_output_name,
                    return_node=True)
                all_q_dq.append(
                    [z_node.layer_name, s_node.layer_name, q_node.layer_name])

        graph.changed_dict[input_node_name] = dict()
        graph.changed_dict[input_node_name]["name"] = output_name
        graph.changed_dict[input_node_name]["total"] = index
        graph.changed_dict[input_node_name]["qdq"] = all_q_dq


@op_mapper('fake_quantize_moving_average_abs_max')
class Fake_quantize_moving_average_abs_max():
    support_opset_version_range = (10, 13)

    @classmethod
    def opset_13(cls, graph, node, **kw):
        zero_node = graph.make_node(
            'Constant', dtype=dtypes.ONNX.INT8, value=[0])

        key = node.input('InScale')
        InScale = None
        for name, param in graph.origin_parameters.items():
            if name in key:
                InScale = param['data']

        input_node_name = node.input('X', 0)

        scale_node = graph.make_node(
            'Constant',
            dtype=dtypes.ONNX.FLOAT,
            value=[float(InScale[0]) / 127.0])

        graph.add_name(node.output('Out', 0))
        output_name = graph.get_name(node.output('Out', 0))

        quantize_node = graph.make_node(
            'QuantizeLinear',
            inputs=[input_node_name, scale_node, zero_node],
            outputs=output_name)

        if input_node_name in graph.changed_dict.keys():
            return

        another_nodes = graph.get_another_node_by_input(
            input_node_name, copy_node=False)
        if len(another_nodes) == 0:
            return

        index = 0
        all_q_dq = list()
        for another_node in another_nodes:
            zero_node, z_node = graph.make_node(
                'Constant', dtype=dtypes.ONNX.INT8, value=[0], return_node=True)
            scale_node, s_node = graph.make_node(
                'Constant',
                dtype=dtypes.ONNX.FLOAT,
                value=[float(InScale[0]) / 127.0],
                return_node=True)
            changed_output_name = output_name + ".paddleadd" + str(index)
            index = index + 1
            quantize_node, q_node = graph.make_node(
                'QuantizeLinear',
                inputs=[input_node_name, scale_node, zero_node],
                return_node=True)
            dequantize_node, dq_node = graph.make_node(
                'DequantizeLinear',
                inputs=[quantize_node, scale_node, zero_node],
                outputs=changed_output_name,
                return_node=True)
            all_q_dq.append([
                z_node.layer_name, s_node.layer_name, q_node.layer_name,
                dq_node.layer_name
            ])

        graph.changed_dict[input_node_name] = dict()
        graph.changed_dict[input_node_name]["name"] = output_name
        graph.changed_dict[input_node_name]["total"] = index
        graph.changed_dict[input_node_name]["qdq"] = all_q_dq


@op_mapper('fake_dequantize_max_abs')
class Fake_dequantize_max_abs():
    support_opset_version_range = (10, 13)

    @classmethod
    def opset_13(cls, graph, node, **kw):
        zero_node = graph.make_node(
            'Constant', dtype=dtypes.ONNX.INT8, value=[0])

        key = node.input('Scale')
        Scale = None
        for name, param in graph.origin_parameters.items():
            if name in key:
                Scale = param['data']

        input_node_name = graph.get_name(node.input('X', 0), with_remove=True)

        scale_node = graph.make_node(
            'Constant',
            dtype=dtypes.ONNX.FLOAT,
            value=[float(Scale[0]) / 127.0])

        graph.add_name(node.output('Out', 0))
        output_name = graph.get_name(node.output('Out', 0))
        # output_name = node.output('Out', 0)

        graph.make_node(
            'DequantizeLinear',
            inputs=[input_node_name, scale_node, zero_node],
            outputs=output_name)


@op_mapper('fake_channel_wise_quantize_abs_max')
class Fake_channel_wise_quantize_abs_max():
    support_opset_version_range = (10, 13)

    @classmethod
    def opset_13(cls, graph, node, **kw):
        paddle.disable_static()
        key = node.input('X')

        update_param = None
        update_name = None
        weight = None
        for name, param in graph.origin_parameters.items():
            if name in key:
                weight = param['data']
                update_name = name
                update_param = param

        tensor = paddle.to_tensor(weight)
        abs_tensor = paddle.abs(tensor)

        scale = None
        input_shape = node.input_shape('X', 0)
        zero_node = None
        vals = None
        if len(input_shape) == 4:
            reshape_tensor = paddle.reshape(
                abs_tensor, shape=[input_shape[0], -1])
            vals, _ = paddle.topk(reshape_tensor, k=1, axis=1)
            scale = vals / 127.0
            scale = paddle.unsqueeze(scale, axis=[2, 3])
            zero_node = graph.make_node(
                'Constant', dtype=dtypes.ONNX.INT8, value=[0] * input_shape[0])

        if len(input_shape) == 2:
            vals, _ = paddle.topk(abs_tensor, k=1, axis=0)
            scale = vals / 127.0
            zero_node = graph.make_node(
                'Constant', dtype=dtypes.ONNX.INT8, value=[0] * input_shape[1])

        div = paddle.divide(tensor, scale)

        numpy_data = np.around(div.numpy())
        round_tensor = paddle.to_tensor(numpy_data)

        round_numpy = np.round(div.numpy())

        clip_tensor = round_tensor
        dq_tensor = paddle.multiply(clip_tensor, scale)

        update_param['data'] = dq_tensor.numpy()
        graph.update_parameters(update_name, update_param)

        scale_numpy = paddle.squeeze(scale).numpy().tolist()

        scale_node = graph.make_node(
            'Constant', dtype=dtypes.ONNX.FLOAT, value=scale_numpy)

        attrs = {'axis': node.attr("quant_axis"), }
        quantize_node = graph.make_node(
            'QuantizeLinear',
            inputs=[node.input('X', 0), scale_node, zero_node],
            outputs=node.output('Out'),
            attrs=attrs)

        key = node.output('OutScale')
        OutScale = None
        update_param = None
        update_name = None
        for name, param in graph.origin_parameters.items():
            if name in key:
                OutScale = param['data']
                update_name = name
                update_param = param

        update_param['data'] = vals.numpy()
        graph.update_parameters(update_name, update_param)


@op_mapper('fake_channel_wise_dequantize_max_abs')
class Fake_channel_wise_dequantize_max_abs():
    support_opset_version_range = (10, 13)

    @classmethod
    def opset_13(cls, graph, node, **kw):
        if graph.static_quantize_model:
            return
        paddle.disable_static()
        key = node.input('Scales', 0)
        InScale1 = None
        if key in graph.origin_parameters.keys():
            param = graph.origin_parameters[key]
            InScale1 = param['data']

        key = node.input('Scales', 1)
        InScale2 = None
        if key is not None and key in graph.origin_parameters.keys():
            param = graph.origin_parameters[key]
            InScale2 = param['data']

        x_num_col_dims = node.attr('x_num_col_dims')
        quant_axis = node.attr('quant_axis')

        if InScale2 is None:
            input_shape = node.input_shape('X', 0)
            channel = input_shape[quant_axis]
            zero_node = graph.make_node(
                'Constant', dtype=dtypes.ONNX.INT8, value=[0] * channel)
            InScale = paddle.to_tensor(InScale1)
            scale = InScale / (127.0)
            scale_numpy = paddle.squeeze(scale).numpy().tolist()
            scale_node = graph.make_node(
                'Constant', dtype=dtypes.ONNX.FLOAT, value=scale_numpy)
            attrs = {'axis': node.attr("quant_axis"), }
            graph.make_node(
                'DequantizeLinear',
                inputs=[node.input('X', 0), scale_node, zero_node],
                outputs=node.output('Out'),
                attrs=attrs)
            return

        if x_num_col_dims > 1:
            input_shape = node.input_shape('X', 0)
            channel = input_shape[x_num_col_dims]
            zero_node = graph.make_node(
                'Constant', dtype=dtypes.ONNX.INT8, value=[0] * channel)
            InScale = paddle.to_tensor(InScale1 * InScale2[0])
            scale = InScale / (127.0 * 127.0)
            scale_numpy = paddle.squeeze(scale).numpy().tolist()
            scale_node = graph.make_node(
                'Constant', dtype=dtypes.ONNX.FLOAT, value=scale_numpy)
            attrs = {'axis': node.attr("x_num_col_dims"), }
            graph.make_node(
                'DequantizeLinear',
                inputs=[node.input('X', 0), scale_node, zero_node],
                outputs=node.output('Out'),
                attrs=attrs)
        else:
            input_shape = node.input_shape('X', 0)
            channel = input_shape[1]
            zero_node = graph.make_node(
                'Constant', dtype=dtypes.ONNX.INT8, value=[0] * channel)
            InScale = paddle.to_tensor(InScale1 * InScale2[0])
            scale = InScale / (127.0 * 127.0)
            scale_numpy = paddle.squeeze(scale).numpy().tolist()
            scale_node = graph.make_node(
                'Constant', dtype=dtypes.ONNX.FLOAT, value=scale_numpy)
            attrs = {'axis': node.attr("x_num_col_dims"), }
            graph.make_node(
                'DequantizeLinear',
                inputs=[node.input('X', 0), scale_node, zero_node],
                outputs=node.output('Out'),
                attrs=attrs)
