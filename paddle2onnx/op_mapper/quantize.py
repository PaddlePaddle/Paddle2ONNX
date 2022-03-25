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
class Quantize_linear():
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

        quantize_node = graph.make_node(
            'QuantizeLinear',
            inputs=[node.input('X', 0), scale_node, zero_node],
            outputs=node.output('Y'),
            axis=node.attr("quant_axis"))


@op_mapper('dequantize_linear')
class Dequantize_linear():
    support_opset_version_range = (13, 15)

    @classmethod
    def opset_13(cls, graph, node, **kw):
        scale_key = node.input('Scale', 0)
        param, weight_scale = mapper_helper.get_param_from_paddle_graph(
            graph, scale_key)
        weight_scale = weight_scale / 127

        scale_list = weight_scale.squeeze().tolist()
        scale_node = graph.make_node(
            'Constant', dtype=dtypes.ONNX.FLOAT, value=scale_list)

        zero_node = graph.make_node(
            'Constant', dtype=dtypes.ONNX.INT8, value=[0] * len(scale_list))

        key = node.input('X', 0)
        update_param, weight = mapper_helper.get_param_from_paddle_graph(graph,
                                                                         key)
        quantize_node = node.input('X', 0)
        if weight is not None:
            if len(weight.shape) == 4:
                new_weight = weight.transpose(1, 2, 3, 0) * weight_scale
                new_weight = new_weight.transpose(3, 0, 1, 2)
            else:
                new_weight = weight * weight_scale
            update_param['data'] = new_weight
            graph.update_parameters(key, update_param)

            quantize_node = graph.make_node(
                'QuantizeLinear',
                inputs=[node.input('X', 0), scale_node, zero_node],
                axis=node.attr("quant_axis"))

        output_name = node.output('Y', 0)
        dequantize_node = graph.make_node(
            'DequantizeLinear',
            inputs=[quantize_node, scale_node, zero_node],
            outputs=[output_name],
            axis=node.attr("quant_axis"))


@op_mapper('fake_quantize_dequantize_moving_average_abs_max')
class Fake_quantize_dequantize_moving_average_abs_max():
    support_opset_version_range = (10, 13)

    @classmethod
    def opset_13(cls, graph, node, **kw):
        zero_node = graph.make_node(
            'Constant', dtype=dtypes.ONNX.INT8, value=[0])

        key = node.input('InScale', 0)
        _, in_scale = mapper_helper.get_param_from_paddle_graph(graph, key)

        input_node_name = node.input('X', 0)

        scale_node = graph.make_node(
            'Constant',
            dtype=dtypes.ONNX.FLOAT,
            value=[float(in_scale[0]) / 127.0])

        graph.add_name(node.output('Out', 0))
        output_name = graph.get_name(node.output('Out', 0))

        quantize_node = graph.make_node(
            'QuantizeLinear', inputs=[input_node_name, scale_node, zero_node])
        graph.make_node(
            'DequantizeLinear',
            inputs=[quantize_node, scale_node, zero_node],
            outputs=output_name)
        if not graph.sortcut_optimize:
            return
        if input_node_name in graph.changed_dict.keys():
            return

        another_nodes = mapper_helper.get_another_node_by_input(
            graph, input_node_name, copy_node=False)
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
                value=[float(in_scale[0]) / 127.0],
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
        quant_axis = node.attr("quant_axis")
        key = node.input('X', 0)
        param, weight = mapper_helper.get_param_from_paddle_graph(graph, key)

        weight_numpy = np.array(weight)
        weight_abs = np.abs(weight_numpy)

        input_shape = node.input_shape('X', 0)
        zero_node = graph.make_node(
            'Constant',
            dtype=dtypes.ONNX.INT8,
            value=[0] * input_shape[quant_axis])
        transposed_weight = weight_abs
        if quant_axis == 1:
            transposed_weight = weight_abs.transpose(1, 0)
        reshaped_weight = np.reshape(transposed_weight,
                                     (input_shape[quant_axis], -1))

        topk_data_sort, _ = mapper_helper.np_topk_helper(
            reshaped_weight, 1, axis=1)
        scale = topk_data_sort / 127.0
        if len(input_shape) == 4:
            scale = np.expand_dims(scale, axis=[2, 3])
        else:
            scale = np.squeeze(scale)

        quantize_weight = weight_numpy / scale
        around_weight_data = np.around(quantize_weight)

        dequantize_weight = around_weight_data * scale

        param['data'] = dequantize_weight
        graph.update_parameters(key, param)

        scale_list = np.squeeze(scale).tolist()

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
        key = node.input('X', 0)
        update_param, weight = mapper_helper.get_param_from_paddle_graph(graph,
                                                                         key)

        weight_numpy = np.array(weight)
        weight_abs = np.abs(weight_numpy)

        zero_node = graph.make_node(
            'Constant', dtype=dtypes.ONNX.INT8, value=[0])

        reshaped_weight = np.reshape(weight_abs, (-1))

        topk_data_sort, _ = mapper_helper.np_topk_helper(
            reshaped_weight, 1, axis=0)
        scale = topk_data_sort / 127.0

        quantize_weight = weight_numpy / scale
        around_weight_data = np.around(quantize_weight)

        dequantize_weight = around_weight_data * scale

        update_param['data'] = dequantize_weight
        graph.update_parameters(key, update_param)

        scale_list = np.squeeze(scale).tolist()

        scale_node = graph.make_node(
            'Constant', dtype=dtypes.ONNX.FLOAT, value=scale_list)

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

        key = node.input('InScale', 0)
        _, in_scale = mapper_helper.get_param_from_paddle_graph(graph, key)

        input_node_name = node.input('X', 0)

        scale_node = graph.make_node(
            'Constant',
            dtype=dtypes.ONNX.FLOAT,
            value=[float(in_scale[0]) / 127.0])

        graph.add_name(node.output('Out', 0))
        output_name = graph.get_name(node.output('Out', 0))

        if graph.quantize_model_mode in ["static"]:
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

        if not graph.sortcut_optimize:
            return

        if input_node_name in graph.changed_dict.keys():
            return

        another_nodes = mapper_helper.get_another_node_by_input(
            graph, input_node_name, copy_node=False)
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
                value=[float(in_scale[0]) / 127.0],
                return_node=True)
            changed_output_name = output_name + ".paddleadd" + str(index)
            index = index + 1
            if graph.quantize_model_mode in ["static"]:
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

        key = node.input('InScale', 0)
        _, in_scale = mapper_helper.get_param_from_paddle_graph(graph, key)

        input_node_name = node.input('X', 0)

        scale_node = graph.make_node(
            'Constant',
            dtype=dtypes.ONNX.FLOAT,
            value=[float(in_scale[0]) / 127.0])

        graph.add_name(node.output('Out', 0))
        output_name = graph.get_name(node.output('Out', 0))

        quantize_node = graph.make_node(
            'QuantizeLinear',
            inputs=[input_node_name, scale_node, zero_node],
            outputs=output_name)
        if not graph.sortcut_optimize:
            return
        if input_node_name in graph.changed_dict.keys():
            return

        another_nodes = mapper_helper.get_another_node_by_input(
            graph, input_node_name, copy_node=False)
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
                value=[float(in_scale[0]) / 127.0],
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

        key = node.input('Scale', 0)
        _, in_scale = mapper_helper.get_param_from_paddle_graph(graph, key)
        in_scale = np.array(in_scale)

        input_node_name = graph.get_name(node.input('X', 0), with_remove=True)

        in_scale = in_scale / 127.0
        in_scale_list = in_scale.tolist()
        scale_node = graph.make_node(
            'Constant', dtype=dtypes.ONNX.FLOAT, value=in_scale_list)

        graph.add_name(node.output('Out', 0))
        output_name = graph.get_name(node.output('Out', 0))

        graph.make_node(
            'DequantizeLinear',
            inputs=[input_node_name, scale_node, zero_node],
            outputs=output_name)


@op_mapper('fake_channel_wise_quantize_abs_max')
class Fake_channel_wise_quantize_abs_max():
    support_opset_version_range = (10, 13)

    @classmethod
    def opset_13(cls, graph, node, **kw):
        quant_axis = node.attr("quant_axis")
        key = node.input('X', 0)

        param, weight = mapper_helper.get_param_from_paddle_graph(graph, key)

        weight_numpy = np.array(weight)
        weight_abs = np.abs(weight_numpy)

        input_shape = node.input_shape('X', 0)
        zero_node = graph.make_node(
            'Constant',
            dtype=dtypes.ONNX.INT8,
            value=[0] * input_shape[quant_axis])
        transposed_weight = weight_abs
        if quant_axis == 1:
            transposed_weight = weight_abs.transpose(1, 0)
        reshaped_weight = np.reshape(transposed_weight,
                                     (input_shape[quant_axis], -1))

        topk_data_sort, _ = mapper_helper.np_topk_helper(
            reshaped_weight, 1, axis=1)
        scale = topk_data_sort / 127.0
        if len(input_shape) == 4:
            scale = np.expand_dims(scale, axis=[2, 3])
        else:
            scale = np.squeeze(scale)

        quantize_weight = weight_numpy / scale
        around_weight_data = np.around(quantize_weight)

        dequantize_weight = around_weight_data * scale

        param['data'] = dequantize_weight
        graph.update_parameters(key, param)

        scale_list = np.squeeze(scale).tolist()

        scale_node = graph.make_node(
            'Constant', dtype=dtypes.ONNX.FLOAT, value=scale_list)

        quantize_node = graph.make_node(
            'QuantizeLinear',
            inputs=[node.input('X', 0), scale_node, zero_node],
            outputs=node.output('Out'),
            axis=quant_axis)

        key = node.output('OutScale', 0)
        update_param, weight = mapper_helper.get_param_from_paddle_graph(graph,
                                                                         key)

        update_param['data'] = vals.numpy()
        graph.update_parameters(key, update_param)


@op_mapper('fake_channel_wise_dequantize_max_abs')
class Fake_channel_wise_dequantize_max_abs():
    support_opset_version_range = (10, 13)

    @classmethod
    def opset_13(cls, graph, node, **kw):
        if graph.quantize_model_mode in ["static"]:
            return

        key = node.input('Scales', 0)
        _, in_scale1 = mapper_helper.get_param_from_paddle_graph(graph, key)

        key = node.input('Scales', 1)
        _, in_scale2 = mapper_helper.get_param_from_paddle_graph(graph, key)

        x_num_col_dims = node.attr('x_num_col_dims')
        quant_axis = node.attr('quant_axis')

        if in_scale2 is None:
            input_shape = node.input_shape('X', 0)
            channel = input_shape[quant_axis]
            zero_node = graph.make_node(
                'Constant', dtype=dtypes.ONNX.INT8, value=[0] * channel)
            in_scale1 = np.array(in_scale1)
            scale = in_scale1 / 127.0
            scale_list = np.squeeze(scale).tolist()
            scale_node = graph.make_node(
                'Constant', dtype=dtypes.ONNX.FLOAT, value=scale_list)
            graph.make_node(
                'DequantizeLinear',
                inputs=[node.input('X', 0), scale_node, zero_node],
                outputs=node.output('Out'),
                axis=quant_axis)
            return

        if x_num_col_dims > 1:
            input_shape = node.input_shape('X', 0)
            channel = input_shape[x_num_col_dims]
            zero_node = graph.make_node(
                'Constant', dtype=dtypes.ONNX.INT8, value=[0] * channel)
            in_scale = np.array(in_scale1 * in_scale2[0])
            scale = in_scale / (127.0 * 127.0)
            scale_list = np.squeeze(scale).tolist()
            scale_node = graph.make_node(
                'Constant', dtype=dtypes.ONNX.FLOAT, value=scale_list)
            graph.make_node(
                'DequantizeLinear',
                inputs=[node.input('X', 0), scale_node, zero_node],
                outputs=node.output('Out'),
                axis=x_num_col_dims)
        else:
            input_shape = node.input_shape('X', 0)
            channel = input_shape[1]
            zero_node = graph.make_node(
                'Constant', dtype=dtypes.ONNX.INT8, value=[0] * channel)
            in_scale = np.array(in_scale1 * in_scale2[0])
            scale = in_scale / (127.0 * 127.0)
            scale_list = np.squeeze(scale).tolist()
            scale_node = graph.make_node(
                'Constant', dtype=dtypes.ONNX.FLOAT, value=scale_list)
            graph.make_node(
                'DequantizeLinear',
                inputs=[node.input('X', 0), scale_node, zero_node],
                outputs=node.output('Out'),
                axis=x_num_col_dims)
