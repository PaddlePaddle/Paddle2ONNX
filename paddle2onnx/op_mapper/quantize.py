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
import paddle
from onnx import helper
import onnx


@op_mapper('fake_quantize_dequantize_moving_average_abs_max')
class Fake_quantize_dequantize_moving_average_abs_max():
    support_opset_version_range = (10, 13)

    @classmethod
    def opset_13(cls, graph, node, **kw):
        print("opset version 13")

        zero_node = graph.make_node(
            'Constant', dtype=dtypes.ONNX.INT8, value=[0])

        key = node.input('InScale')
        InScale = None
        for name, param in graph.origin_parameters.items():
            if name in key:
                InScale = param['data']

        Minus_InScale = -1.0 * InScale
        print("name, minus_scale, inscale: ", key, Minus_InScale, InScale)

        clip_node = node.input('X', 0)

        scale_node = graph.make_node(
            'Constant',
            dtype=dtypes.ONNX.FLOAT,
            value=[float(InScale[0]) / 127.0])

        quantize_node = graph.make_node(
            'QuantizeLinear', inputs=[clip_node, scale_node, zero_node])
        graph.make_node(
            'DequantizeLinear',
            inputs=[quantize_node, scale_node, zero_node],
            outputs=node.output('Out'))


@op_mapper('fake_channel_wise_quantize_dequantize_abs_max')
class Fake_channel_wise_quantize_dequantize_abs_max():
    support_opset_version_range = (10, 13)

    @classmethod
    def opset_13(cls, graph, node, **kw):
        print("opset version 10")
        paddle.disable_static()
        key = node.input('X')

        update_param = None
        update_name = None
        weight = None
        for name, param in graph.origin_parameters.items():
            if name in key:
                print(name)
                weight = param['data']
                update_name = name
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

        # print("vals: ",vals)
        # tensor = paddle.clip(tensor, min=-127.0, max=127.0)
        div = paddle.divide(tensor, scale)

        numpy_data = np.around(div.numpy())
        round_tensor = paddle.to_tensor(numpy_data)

        round_numpy = np.round(div.numpy())
        print("max diff: ", np.amax(abs(round_numpy - numpy_data)))
        # print("diff: ", round_numpy - numpy_data)

        # clip_tensor = paddle.clip(round_tensor, min=-127.0, max=127.0)
        clip_tensor = round_tensor
        dq_tensor = paddle.multiply(clip_tensor, scale)
        # print("q_dq tensor: ",dq_tensor)

        # print("q_dq_q tensor: ",dq_tensor / scale)

        update_param['data'] = dq_tensor.numpy()
        graph.update_parameters(update_name, update_param)

        scale_numpy = paddle.squeeze(scale).numpy().tolist()
        # print("scale_numpy: ",scale_numpy)
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

    @classmethod
    def opset_10(cls, graph, node, **kw):
        print("opset version 13")

        abs_node = graph.make_node('Abs', inputs=node.input('X', 0))

        input_shape = node.input_shape('X', 0)
        if len(input_shape) == 4:
            one_node = graph.make_node(
                'Constant', dtype=dtypes.ONNX.INT64, value=[1])

            shape_node = graph.make_node(
                'Constant', dtype=dtypes.ONNX.INT64,
                value=[input_shape[0], -1])
            reshaped_node = graph.make_node(
                'Reshape', inputs=[abs_node, shape_node])

            topk_node = graph.make_node(
                'TopK', inputs=[reshaped_node, one_node], outputs=2, axis=1)

            val_node = topk_node[0]

            # minus_node = graph.make_node(
            #     'Constant', dtype=dtypes.ONNX.FLOAT, value=[-1.])
            # minus_val_node = graph.make_node(
            #     'Mul', inputs=[val_node, minus_node])

            # clip_node = graph.make_node(
            #     'Clip', inputs=[node.input('X',0), minus_val_node, val_node])
            # clip_node =

            bins_node = graph.make_node(
                'Constant', dtype=dtypes.ONNX.FLOAT, value=127.)
            scale_node = graph.make_node('Div', inputs=[val_node, bins_node])

            shape_node = graph.make_node(
                'Constant', dtype=dtypes.ONNX.INT64, value=[-1])
            scale_node = graph.make_node(
                'Reshape', inputs=[scale_node, shape_node])

            zero_node = graph.make_node(
                'Constant', dtype=dtypes.ONNX.INT8, value=[0] * input_shape[0])

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

        if len(input_shape) == 2:
            print("no impletment!")


@op_mapper('moving_average_abs_max_scale')
class Moving_average_abs_max_scale():
    support_opset_version_range = (10, 13)

    @classmethod
    def opset_13(cls, graph, node, **kw):
        print("opset version 13")
        zero_node = graph.make_node(
            'Constant', dtype=dtypes.ONNX.FLOAT, value=[0.])

        graph.make_node(
            'Add',
            inputs=[node.input('X', 0), zero_node],
            outputs=node.output('Out'))
