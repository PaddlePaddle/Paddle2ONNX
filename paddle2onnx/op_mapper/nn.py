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


@op_mapper(['conv2d', 'depthwise_conv2d'])
class Conv():
    support_opset_verison_range = (1, 12)

    @classmethod
    def opset_1(cls, graph, node, **kw):
        kernel_shape = node.input_shape('Filter', 0)
        graph.make_node(
            'Conv',
            inputs=node.input('Input') + node.input('Filter'),
            outputs=node.output('Output'),
            dilations=node.attr('dilations'),
            kernel_shape=kernel_shape[-2:],
            strides=node.attr('strides'),
            group=node.attr('groups'),
            pads=node.attr('paddings') + node.attr('paddings'))


@op_mapper('conv2d_transpose')
class ConvTranspose():
    support_opset_verison_range = (1, 12)

    @classmethod
    def opset_1(cls, graph, node, **kw):
        kernel_shape = node.input_shape('Filter', 0)
        node = graph.make_node(
            'ConvTranspose',
            inputs=node.input('Input') + node.input('Filter'),
            outputs=node.output('Output'),
            dilations=node.attr('dilations'),
            kernel_shape=kernel_shape[-2:],
            strides=node.attr('strides'),
            group=1,
            pads=node.attr('paddings') + node.attr('paddings'))


@op_mapper('pool2d')
class Pool():
    support_opset_verison_range = (1, 12)
    pool_type = {
        'max': ('MaxPool', 'GlobalMaxPool'),
        'avg': ('AveragePool', 'GlobalAveragePool')
    }

    @classmethod
    def opset_1(cls, graph, node, **kw):
        if node.attr('global_pooling'):
            onnx_node = graph.make_node(
                cls.pool_type[node.attr('pooling_type')][1],
                inputs=node.input('X'),
                outputs=node.output('Out'))
        elif node.attr('adaptive'):
            input_height, input_width = node.input_shape('X', 0)[2:]
            output_height, output_width = node.output_shape('Out', 0)[2:]
            if input_height % output_height != 0 or input_width % output_width != 0:
                raise Exception("ONNX cannot support adaptive pool")
            else:
                stride_h = int(input_height / output_height)
                stride_w = int(input_width / output_width)
                kernel_h = input_height - (output_height - 1) * stride_h
                kernel_w = input_width - (output_width - 1) * stride_w

                onnx_node = graph.make_node(
                    cls.pool_type[node.attr('pooling_type')][0],
                    inputs=node.input('X'),
                    outputs=node.output('Out'),
                    kernel_shape=(kernel_h, kernel_w),
                    strides=(stride_h, stride_w),
                    pads=[0, 0, 0, 0])
        else:
            input_shape = node.input_shape('X', 0)
            k_size = node.attr('ksize')
            paddings = node.attr('paddings')
            if input_shape[2] > 0 and input_shape[2] + paddings[0] < k_size[0]:
                k_size[0] = input_shape[2] + paddings[0]
            if input_shape[3] > 0 and input_shape[3] + paddings[1] < k_size[1]:
                k_size[1] = input_shape[3] + paddings[1]
            onnx_node = graph.make_node(
                cls.pool_type[node.attr('pooling_type')][0],
                inputs=node.input('X'),
                outputs=node.output('Out'),
                kernel_shape=k_size,
                strides=node.attr('strides'),
                pads=node.attr('paddings') + node.attr('paddings'))


@op_mapper('norm')
class Norm():
    support_opset_verison_range = (1, 12)

    @classmethod
    def opset_1(cls, graph, node, **kw):
        node = graph.make_node(
            'LpNormalization',
            inputs=node.input('X'),
            outputs=node.output('Out'),
            axis=node.attr('axis'))


@op_mapper('batch_norm')
class BatchNorm():
    support_opset_verison_range = (1, 12)

    @classmethod
    def make_attrs_and_inputs(cls, graph, node, **kw):
        onnx_attr = {
            'epsilon': node.attr('epsilon'),
            'momentum': node.attr('momentum')
        }
        inputs = node.input('X') + node.input('Scale') + node.input(
            'Bias') + node.input('Mean') + node.input('Variance')
        return onnx_attr, inputs

    @classmethod
    def opset_9(cls, graph, node, **kw):
        onnx_attr, inputs = cls.make_attrs_and_inputs(graph, node, **kw)
        onnx_node = graph.make_node(
            'BatchNormalization',
            inputs=inputs,
            outputs=node.output('Y'),
            **onnx_attr)

    @classmethod
    def opset_7(cls, graph, node, **kw):
        onnx_attr, inputs = cls.make_attrs_and_inputs(graph, node, **kw)
        onnx_attr['spatial'] = 0
        onnx_node = graph.make_node(
            'BatchNormalization',
            inputs=inputs,
            outputs=node.output('Y'),
            **onnx_attr)

    @classmethod
    def opset_1(cls, graph, node, **kw):
        onnx_attr, inputs = cls.make_attrs_and_inputs(graph, node, **kw)
        onnx_attr['is_test'] = 1
        onnx_node = graph.make_node(
            'BatchNormalization',
            inputs=inputs,
            outputs=node.output('Y'),
            **onnx_attr)


@op_mapper('instance_norm')
class InstanceNorm():
    support_opset_verison_range = (1, 12)

    @classmethod
    def opset_1(cls, graph, node, **kw):
        onnx_attr = {'epsilon': node.attr('epsilon'), }
        inputs = node.input('X') + node.input('Scale') + node.input('Bias')
        onnx_node = graph.make_node(
            'InstanceNormalization',
            inputs=inputs,
            outputs=node.output('Y'),
            **onnx_attr)


@op_mapper('dropout')
class Dropout():
    support_opset_verison_range = (7, 12)

    @classmethod
    def opset_7(cls, graph, node, **kw):
        dropout_mode = node.attr('dropout_implementation')
        dropout_prob = node.attr('dropout_prob')
        if dropout_mode == 'upscale_in_train':
            onnx_node = graph.make_node(
                'Identity', inputs=node.input('X'), outputs=node.output('Out'))
        elif dropout_mode == 'downgrade_in_infer':
            scale_node = graph.make_node(
                'Constant',
                attrs={'dtype': dtypes.ONNX.FLOAT,
                       'value': 1 - dropout_prob})
            graph.make_node(
                "Mul",
                inputs=[node.input('X')[0], scale_node],
                outputs=node.output('Out'))
        else:
            raise Exception("Unexpected situation happend")


@op_mapper('roi_align')
class RoiAlign():
    support_opset_verison_range = (10, 12)

    @classmethod
    def opset_10(cls, graph, node, **kw):
        rois_shape = graph.make_node('Shape', inputs=[node.input('ROIs', 0)])
        starts = graph.make_node(
            'Constant', attrs={'dtype': dtypes.ONNX.INT64,
                               'value': [0]})
        ends = graph.make_node(
            'Constant', attrs={'dtype': dtypes.ONNX.INT64,
                               'value': [1]})
        num_rois = graph.make_node('Slice', inputs=[rois_shape, starts, ends])
        zero = graph.make_node(
            'Constant', dims=[1], dtype=dtypes.ONNX.INT64, value=[0])
        batch_indices = graph.make_node('Expand', inputs=[zero, num_rois])
        node = graph.make_node(
            'RoiAlign',
            inputs=[node.input('X', 0), node.input('ROIs', 0), batch_indices],
            outputs=node.output('Out'),
            mode='avg',
            output_height=node.attr('pooled_height'),
            output_width=node.attr('pooled_width'),
            sampling_ratio=node.attr('sampling_ratio'),
            spatial_scale=node.attr('spatial_scale'))
