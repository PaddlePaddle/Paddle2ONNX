#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import paddle2onnx as p2o
from paddle2onnx.utils import logging
from paddle2onnx.constant import dtypes


class ResizeForTensorRT():
    support_opset_verison_range = (10, 12)

    @classmethod
    def opset_10(cls, graph, node, **kw):
        inputs = [node.input('X')[0]]
        logging.warning(
            "When convert paddle node {},  Resize is converted to asytmmetric floor neareset mode for TensorRT.".
            format(node.__str__()))
        if len(node.input('OutSize')) > 0 or len(node.input('SizeTensor')) > 0:
            in_shape, out_shape = cls.compute_output_shape(graph, node)
            cast_shape_node2 = graph.make_node(
                'Cast', inputs=[out_shape], to=dtypes.ONNX.FLOAT)
            cast_shape_node0 = graph.make_node(
                'Cast', inputs=[in_shape], to=dtypes.ONNX.FLOAT)
            node_h_w_scales = graph.make_node(
                'Div', inputs=[cast_shape_node2, cast_shape_node0])
            inputs.append(node_h_w_scales)
        elif 'Scale' in node.inputs and len(node.input('Scale')) > 0:
            scale = node.input('Scale')[0]
            inputs.append(scale)
        else:
            out_shape = [node.attr('out_h'), node.attr('out_w')]
            scale = node.attr('scale')
            if isinstance(scale, float):
                scale = [1, 1, scale, scale]
            else:
                scale = [1, 1] + scale
            if out_shape.count(-1) > 0:
                scale_node = graph.make_node(
                    'Constant',
                    attrs={'dtype': dtypes.ONNX.FLOAT,
                           'value': scale})
                inputs.append(scale_node)
            else:
                raise Exception("Unexpected situation happend")
        graph.make_node(
            'Resize',
            inputs=inputs,
            outputs=node.output('Out'),
            mode=resize_type)

    @classmethod
    def opset_11(cls, graph, node, **kw):
        node_lists = []
        logging.warning(
            "When convert paddle node {},  Resize is converted to asytmmetric floor neareset mode for TensorRT.".
            format(node.__str__()))
        roi_node = graph.make_node(
            'Constant',
            attrs={
                'dtype': dtypes.ONNX.FLOAT,
                'value': [1, 1, 1, 1, 1, 1, 1, 1]
            })
        inputs = [node.input('X')[0], roi_node]
        node_lists.append(roi_node)
        if len(node.input('OutSize')) > 0 or len(node.input('SizeTensor')) > 0:
            empty_node = graph.make_node(
                'Constant', attrs={'dtype': dtypes.ONNX.FLOAT,
                                   'value': []})
            inputs.append(empty_node)
            _, out_shape = cls.compute_output_shape(graph, node)
            inputs.append(out_shape)
        elif len(node.input('Scale')) > 0:
            scale = node.input('Scale')[0]
            inputs.append(scale)
        else:
            out_shape = [node.attr('out_h'), node.attr('out_w')]
            scale = node.attr('scale')
            if isinstance(scale, float):
                scale = [1, 1, scale, scale]
            else:
                scale = [1, 1] + scale

            if out_shape.count(-1) > 0:
                scale_node = graph.make_node(
                    'Constant',
                    attrs={'dtype': dtypes.ONNX.FLOAT,
                           'value': scale})
                inputs.append(scale_node)
            else:
                empty_node = graph.make_node(
                    'Constant',
                    attrs={'dtype': dtypes.ONNX.FLOAT,
                           'value': []})
                in_shape, out_shape = cls.compute_output_shape_by_size(graph,
                                                                       node)
                inputs += [empty_node, out_shape]
        graph.make_node(
            'Resize',
            inputs=inputs,
            outputs=node.output('Out'),
            coordinate_transformation_mode='asymmetric',
            mode='nearest',
            nearest_mode='floor')


def register_resize_for_tensorrt():
    paddle_resize_type = [
        'bilinear_interp', 'nearest_interp', 'bilinear_interp_v2',
        'nearest_interp_v2'
    ]
    p2o.register_op_mapper(paddle_resize_type, ResizeForTensorRT)
