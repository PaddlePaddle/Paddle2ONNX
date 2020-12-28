# Copyright (c) 2020  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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

MAX_FLOAT32 = np.asarray(
    [255, 255, 127, 127], dtype=np.uint8).view(np.float32)[0]


def is_static_shape(shape):
    if len(shape) > 1 and shape.count(-1) > 1:
        raise Exception(
            "Converting this model to ONNX need with static input shape," \
            " please fix input shape of this model, see doc Q2 in" \
            " https://github.com/PaddlePaddle/paddle2onnx/blob/develop/FAQ.md."
        )


@op_mapper('yolo_box')
class YOLOBox():
    support_opset_verison_range = (9, 12)

    node_pred_box_x1_decode = None
    node_pred_box_y1_decode = None
    node_pred_box_x2_decode = None
    node_pred_box_y2_decode = None
    node_pred_box_x2_sub_w = None
    node_pred_box_y2_sub_h = None

    @classmethod
    def front(cls, graph, node, **kw):
        model_name = node.output('Boxes', 0)
        input_shape = node.input_shape('X', 0)
        is_static_shape(input_shape)
        image_size = node.input('ImgSize')
        input_height = input_shape[2]
        input_width = input_shape[3]
        class_num = node.attr('class_num')
        anchors = node.attr('anchors')
        num_anchors = int(len(anchors)) // 2
        downsample_ratio = node.attr('downsample_ratio')
        input_size = input_height * downsample_ratio
        conf_thresh = node.attr('conf_thresh')
        conf_thresh_mat = [conf_thresh
                           ] * num_anchors * input_height * input_width

        cls.score_shape = [
            1, input_height * input_width * int(num_anchors), class_num
        ]

        im_outputs = []

        x_shape = [1, num_anchors, 5 + class_num, input_height, input_width]
        node_x_shape = graph.make_node(
            'Constant', attrs={'dtype': dtypes.ONNX.INT64,
                               'value': x_shape})

        node_x_reshape = graph.make_node(
            'Reshape', inputs=[node.input('X')[0], node_x_shape])
        node_x_transpose = graph.make_node(
            'Transpose', inputs=[node_x_reshape], perm=[0, 1, 3, 4, 2])

        range_x = []
        range_y = []
        for i in range(0, input_width):
            range_x.append(i)
        for j in range(0, input_height):
            range_y.append(j)

        node_range_x = graph.make_node(
            'Constant', attrs={
                'dtype': dtypes.ONNX.FLOAT,
                'value': range_x,
            })

        node_range_y = graph.make_node(
            'Constant',
            inputs=[],
            attrs={
                'dtype': dtypes.ONNX.FLOAT,
                'value': range_y,
            })

        range_x_new_shape = [1, input_width]
        range_y_new_shape = [input_height, 1]

        node_range_x_new_shape = graph.make_node(
            'Constant',
            inputs=[],
            dtype=dtypes.ONNX.INT64,
            value=range_x_new_shape)
        node_range_y_new_shape = graph.make_node(
            'Constant',
            inputs=[],
            dtype=dtypes.ONNX.INT64,
            value=range_y_new_shape)

        node_range_x_reshape = graph.make_node(
            'Reshape', inputs=[node_range_x, node_range_x_new_shape])
        node_range_y_reshape = graph.make_node(
            'Reshape', inputs=[node_range_y, node_range_y_new_shape])

        node_grid_x = graph.make_node(
            "Tile", inputs=[node_range_x_reshape, node_range_y_new_shape])

        node_grid_y = graph.make_node(
            "Tile", inputs=[node_range_y_reshape, node_range_x_new_shape])

        node_box_x = model_name + "@box_x"
        node_box_y = model_name + "@box_y"
        node_box_w = model_name + "@box_w"
        node_box_h = model_name + "@box_h"
        node_conf = model_name + "@conf"
        node_prob = model_name + "@prob"

        node_split_input = graph.make_node(
            "Split",
            inputs=[node_x_transpose],
            outputs=[
                node_box_x, node_box_y, node_box_w, node_box_h, node_conf,
                node_prob
            ],
            axis=-1,
            split=[1, 1, 1, 1, 1, class_num])

        node_box_x_sigmoid = graph.make_node("Sigmoid", inputs=[node_box_x])

        node_box_y_sigmoid = graph.make_node("Sigmoid", inputs=[node_box_y])

        node_box_x_squeeze = graph.make_node(
            'Squeeze', inputs=[node_box_x_sigmoid], axes=[4])

        node_box_y_squeeze = graph.make_node(
            'Squeeze', inputs=[node_box_y_sigmoid], axes=[4])

        node_box_x_add_grid = graph.make_node(
            "Add", inputs=[node_grid_x, node_box_x_squeeze])

        node_box_y_add_grid = graph.make_node(
            "Add", inputs=[node_grid_y, node_box_y_squeeze])

        node_input_h = graph.make_node(
            'Constant',
            inputs=[],
            dtype=dtypes.ONNX.FLOAT,
            value=[input_height])

        node_input_w = graph.make_node(
            'Constant', inputs=[], dtype=dtypes.ONNX.FLOAT,
            value=[input_width])

        node_box_x_encode = graph.make_node(
            'Div', inputs=[node_box_x_add_grid, node_input_w])

        node_box_y_encode = graph.make_node(
            'Div', inputs=[node_box_y_add_grid, node_input_h])

        node_anchor_tensor = graph.make_node(
            "Constant", inputs=[], dtype=dtypes.ONNX.FLOAT, value=anchors)

        anchor_shape = [int(num_anchors), 2]
        node_anchor_shape = graph.make_node(
            "Constant", inputs=[], dtype=dtypes.ONNX.INT64, value=anchor_shape)

        node_anchor_tensor_reshape = graph.make_node(
            "Reshape", inputs=[node_anchor_tensor, node_anchor_shape])

        node_input_size = graph.make_node(
            "Constant", inputs=[], dtype=dtypes.ONNX.FLOAT, value=[input_size])

        node_anchors_div_input_size = graph.make_node(
            "Div", inputs=[node_anchor_tensor_reshape, node_input_size])

        node_anchor_w = model_name + "@anchor_w"
        node_anchor_h = model_name + "@anchor_h"

        node_anchor_split = graph.make_node(
            'Split',
            inputs=[node_anchors_div_input_size],
            outputs=[node_anchor_w, node_anchor_h],
            axis=1,
            split=[1, 1])

        new_anchor_shape = [1, int(num_anchors), 1, 1]
        node_new_anchor_shape = graph.make_node(
            'Constant',
            inputs=[],
            dtype=dtypes.ONNX.INT64,
            value=new_anchor_shape)

        node_anchor_w_reshape = graph.make_node(
            'Reshape', inputs=[node_anchor_w, node_new_anchor_shape])

        node_anchor_h_reshape = graph.make_node(
            'Reshape', inputs=[node_anchor_h, node_new_anchor_shape])

        node_box_w_squeeze = graph.make_node(
            'Squeeze', inputs=[node_box_w], axes=[4])

        node_box_h_squeeze = graph.make_node(
            'Squeeze', inputs=[node_box_h], axes=[4])

        node_box_w_exp = graph.make_node("Exp", inputs=[node_box_w_squeeze])
        node_box_h_exp = graph.make_node("Exp", inputs=[node_box_h_squeeze])

        node_box_w_encode = graph.make_node(
            'Mul', inputs=[node_box_w_exp, node_anchor_w_reshape])

        node_box_h_encode = graph.make_node(
            'Mul', inputs=[node_box_h_exp, node_anchor_h_reshape])

        node_conf_sigmoid = graph.make_node('Sigmoid', inputs=[node_conf])

        node_conf_thresh = graph.make_node(
            'Constant',
            inputs=[],
            dtype=dtypes.ONNX.FLOAT,
            value=conf_thresh_mat)

        conf_shape = [1, int(num_anchors), input_height, input_width, 1]
        node_conf_shape = graph.make_node(
            'Constant', inputs=[], dtype=dtypes.ONNX.INT64, value=conf_shape)

        node_conf_thresh_reshape = graph.make_node(
            'Reshape', inputs=[node_conf_thresh, node_conf_shape])

        node_conf_sub = graph.make_node(
            'Sub', inputs=[node_conf_sigmoid, node_conf_thresh_reshape])

        node_conf_clip = graph.make_node('Clip', inputs=[node_conf_sub])

        zeros = [0]
        node_zeros = graph.make_node(
            'Constant', inputs=[], dtype=dtypes.ONNX.FLOAT, value=zeros)

        node_conf_clip_bool = graph.make_node(
            'Greater', inputs=[node_conf_clip, node_zeros])

        node_conf_clip_cast = graph.make_node(
            'Cast', inputs=[node_conf_clip_bool], to=1)

        node_conf_set_zero = graph.make_node(
            'Mul', inputs=[node_conf_sigmoid, node_conf_clip_cast])

        node_prob_sigmoid = graph.make_node('Sigmoid', inputs=[node_prob])

        new_shape = [1, int(num_anchors), input_height, input_width, 1]
        node_new_shape = graph.make_node(
            'Constant',
            inputs=[],
            dtype=dtypes.ONNX.INT64,
            dims=[len(new_shape)],
            value=new_shape)

        node_conf_new_shape = graph.make_node(
            'Reshape', inputs=[node_conf_set_zero, node_new_shape])

        cls.node_score = graph.make_node(
            'Mul', inputs=[node_prob_sigmoid, node_conf_new_shape])

        node_conf_bool = graph.make_node(
            'Greater', inputs=[node_conf_new_shape, node_zeros])

        node_box_x_new_shape = graph.make_node(
            'Reshape', inputs=[node_box_x_encode, node_new_shape])

        node_box_y_new_shape = graph.make_node(
            'Reshape', inputs=[node_box_y_encode, node_new_shape])

        node_box_w_new_shape = graph.make_node(
            'Reshape', inputs=[node_box_w_encode, node_new_shape])

        node_box_h_new_shape = graph.make_node(
            'Reshape', inputs=[node_box_h_encode, node_new_shape])

        node_pred_box = graph.make_node(
            'Concat',
            inputs=[node_box_x_new_shape, node_box_y_new_shape, \
                   node_box_w_new_shape, node_box_h_new_shape],
            axis=4)

        node_conf_cast = graph.make_node('Cast', inputs=[node_conf_bool], to=1)

        node_pred_box_mul_conf = graph.make_node(
            'Mul', inputs=[node_pred_box, node_conf_cast])

        box_shape = [1, int(num_anchors) * input_height * input_width, 4]
        node_box_shape = graph.make_node(
            'Constant', inputs=[], dtype=dtypes.ONNX.INT64, value=box_shape)

        node_pred_box_new_shape = graph.make_node(
            'Reshape', inputs=[node_pred_box_mul_conf, node_box_shape])

        node_pred_box_x = model_name + "@_pred_box_x"
        node_pred_box_y = model_name + "@_pred_box_y"
        node_pred_box_w = model_name + "@_pred_box_w"
        node_pred_box_h = model_name + "@_pred_box_h"

        node_pred_box_split = graph.make_node(
            'Split',
            inputs=[node_pred_box_new_shape],
            outputs=[
                node_pred_box_x, node_pred_box_y, node_pred_box_w,
                node_pred_box_h
            ],
            axis=2)

        node_number_two = graph.make_node(
            "Constant", inputs=[], dtype=dtypes.ONNX.FLOAT, value=[2])

        node_half_w = graph.make_node(
            "Div", inputs=[node_pred_box_w, node_number_two])

        node_half_h = graph.make_node(
            "Div", inputs=[node_pred_box_h, node_number_two])

        node_pred_box_x1 = graph.make_node(
            'Sub', inputs=[node_pred_box_x, node_half_w])

        node_pred_box_y1 = graph.make_node(
            'Sub', inputs=[node_pred_box_y, node_half_h])

        node_pred_box_x2 = graph.make_node(
            'Add', inputs=[node_pred_box_x, node_half_w])

        node_pred_box_y2 = graph.make_node(
            'Add', inputs=[node_pred_box_y, node_half_h])

        node_sqeeze_image_size = graph.make_node(
            "Squeeze", inputs=image_size, axes=[0])

        node_img_height = model_name + "@img_height"
        node_img_width = model_name + "@img_width"
        node_image_size_split = graph.make_node(
            "Split",
            inputs=[node_sqeeze_image_size],
            outputs=[node_img_height, node_img_width],
            axis=-1,
            split=[1, 1])

        node_img_width_cast = graph.make_node(
            'Cast', inputs=[node_img_width], to=1)

        node_img_height_cast = graph.make_node(
            'Cast', inputs=[node_img_height], to=1)

        cls.node_pred_box_x1_decode = graph.make_node(
            'Mul', inputs=[node_pred_box_x1, node_img_width_cast])

        cls.node_pred_box_y1_decode = graph.make_node(
            'Mul', inputs=[node_pred_box_y1, node_img_height_cast])

        cls.node_pred_box_x2_decode = graph.make_node(
            'Mul', inputs=[node_pred_box_x2, node_img_width_cast])

        cls.node_pred_box_y2_decode = graph.make_node(
            'Mul', inputs=[node_pred_box_y2, node_img_height_cast])

        node_number_one = graph.make_node(
            'Constant', inputs=[], dtype=dtypes.ONNX.FLOAT, value=[1])

        node_new_img_height = graph.make_node(
            'Sub', inputs=[node_img_height_cast, node_number_one])

        node_new_img_width = graph.make_node(
            'Sub', inputs=[node_img_width_cast, node_number_one])

        cls.node_pred_box_x2_sub_w = graph.make_node(
            'Sub', inputs=[cls.node_pred_box_x2_decode, node_new_img_width])

        cls.node_pred_box_y2_sub_h = graph.make_node(
            'Sub', inputs=[cls.node_pred_box_y2_decode, node_new_img_height])

    @classmethod
    def opset_9(cls, graph, node, **kw):
        cls.front(graph, node, **kw)
        node_pred_box_x1_clip = graph.make_node(
            'Clip',
            inputs=[cls.node_pred_box_x1_decode],
            min=0.0,
            max=float(MAX_FLOAT32))

        node_pred_box_y1_clip = graph.make_node(
            'Clip',
            inputs=[cls.node_pred_box_y1_decode],
            min=0.0,
            max=float(MAX_FLOAT32))

        node_pred_box_x2_clip = graph.make_node(
            'Clip',
            inputs=[cls.node_pred_box_x2_sub_w],
            min=0.0,
            max=float(MAX_FLOAT32))

        node_pred_box_y2_clip = graph.make_node(
            'Clip',
            inputs=[cls.node_pred_box_y2_sub_h],
            min=0.0,
            max=float(MAX_FLOAT32))

        node_pred_box_x2_res = graph.make_node(
            'Sub', inputs=[cls.node_pred_box_x2_decode, node_pred_box_x2_clip])

        node_pred_box_y2_res = graph.make_node(
            'Sub', inputs=[cls.node_pred_box_y2_decode, node_pred_box_y2_clip])

        node_pred_box_result = graph.make_node(
            'Concat',
            inputs=[
                node_pred_box_x1_clip, node_pred_box_y1_clip,
                node_pred_box_x2_res, node_pred_box_y2_res
            ],
            outputs=node.output('Boxes'),
            axis=-1)

        node_score_shape = graph.make_node(
            "Constant",
            inputs=[],
            dtype=dtypes.ONNX.INT64,
            value=cls.score_shape)

        node_score_new_shape = graph.make_node(
            'Reshape',
            inputs=[cls.node_score, node_score_shape],
            outputs=node.output('Scores'))

    @classmethod
    def opset_11(cls, graph, node, **kw):
        cls.front(graph, node, **kw)
        min_const = graph.make_node(
            'Constant', inputs=[], dtype=dtypes.ONNX.FLOAT, value=0.0)

        max_const = graph.make_node(
            'Constant', inputs=[], dtype=dtypes.ONNX.FLOAT, value=MAX_FLOAT32)

        node_pred_box_x1_clip = graph.make_node(
            'Clip', inputs=[cls.node_pred_box_x1_decode, min_const, max_const])

        node_pred_box_y1_clip = graph.make_node(
            'Clip', inputs=[cls.node_pred_box_y1_decode, min_const, max_const])

        node_pred_box_x2_clip = graph.make_node(
            'Clip', inputs=[cls.node_pred_box_x2_sub_w, min_const, max_const])

        node_pred_box_y2_clip = graph.make_node(
            'Clip', inputs=[cls.node_pred_box_y2_sub_h, min_const, max_const])

        node_pred_box_x2_res = graph.make_node(
            'Sub', inputs=[cls.node_pred_box_x2_decode, node_pred_box_x2_clip])

        node_pred_box_y2_res = graph.make_node(
            'Sub', inputs=[cls.node_pred_box_y2_decode, node_pred_box_y2_clip])

        node_pred_box_result = graph.make_node(
            'Concat',
            inputs=[
                node_pred_box_x1_clip, node_pred_box_y1_clip,
                node_pred_box_x2_res, node_pred_box_y2_res
            ],
            outputs=node.output('Boxes'),
            axis=-1)

        node_score_shape = graph.make_node(
            "Constant",
            inputs=[],
            dtype=dtypes.ONNX.INT64,
            value=cls.score_shape)

        node_score_new_shape = graph.make_node(
            'Reshape',
            inputs=[cls.node_score, node_score_shape],
            outputs=node.output('Scores'))
