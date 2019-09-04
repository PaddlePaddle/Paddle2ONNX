# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

import sys
import math
import onnx
import warnings
import numpy as np
from functools import partial
from onnx import TensorProto
from onnx.helper import make_node, make_tensor
from onnx import onnx_pb
from paddle.fluid.executor import _fetch_var as fetch_var
from fluid.utils import op_io_info, get_old_name
from fluid_onnx.variables import PADDLE_TO_ONNX_DTYPE, paddle_onnx_shape
from onnx import helper
import paddle.fluid.core as core


def yolo_box_op(operator, block):
    """
        Convert the paddle yolo_box op to onnx op.
        This op is to decode output of yolo.
    """
    inputs, attrs, outputs = op_io_info(operator)
    model_name = outputs['Boxes'][0]
    input_shape = block.vars[get_old_name(inputs['X'][0])].shape
    image_size = inputs['ImgSize']
    input_height = input_shape[2]
    input_width = input_shape[3]

    class_num = attrs['class_num']
    anchors = attrs['anchors']
    num_anchors = int(len(anchors)) // 2
    downsample_ratio = attrs['downsample_ratio']
    input_size = input_height * downsample_ratio
    conf_thresh = attrs['conf_thresh']
    conf_thresh_mat = np.ones(
        [num_anchors * input_height * input_width]) * conf_thresh

    node_list = []
    im_outputs = []

    x_shape = [1, num_anchors, 5 + class_num, input_height, input_width]
    name_x_shape = [model_name + "@x_shape"]
    node_x_shape = onnx.helper.make_node(
        'Constant',
        inputs=[],
        outputs=name_x_shape,
        value=onnx.helper.make_tensor(
            name=name_x_shape[0] + "@const",
            data_type=onnx.TensorProto.INT64,
            dims=[5],
            vals=x_shape))
    node_list.append(node_x_shape)

    outputs_x_reshape = [model_name + "@reshape"]
    node_x_reshape = onnx.helper.make_node(
        'Reshape', inputs=inputs['X'] + name_x_shape, outputs=outputs_x_reshape)
    node_list.append(node_x_reshape)

    outputs_x_transpose = [model_name + "@x_transpose"]
    node_x_transpose = onnx.helper.make_node(
        'Transpose',
        inputs=outputs_x_reshape,
        outputs=outputs_x_transpose,
        perm=[0, 1, 3, 4, 2])
    node_list.append(node_x_transpose)

    range_x = []
    range_y = []
    for i in range(0, input_width):
        range_x.append(i)
    for j in range(0, input_height):
        range_y.append(j)

    name_range_x = [model_name + "@range_x"]
    node_range_x = onnx.helper.make_node(
        'Constant',
        inputs=[],
        outputs=name_range_x,
        value=onnx.helper.make_tensor(
            name=name_range_x[0] + "@const",
            data_type=onnx.TensorProto.FLOAT,
            dims=[input_width],
            vals=range_x))
    node_list.append(node_range_x)

    name_range_y = [model_name + "@range_y"]
    node_range_y = onnx.helper.make_node(
        'Constant',
        inputs=[],
        outputs=name_range_y,
        value=onnx.helper.make_tensor(
            name=name_range_y[0] + "@const",
            data_type=onnx.TensorProto.FLOAT,
            dims=[input_height],
            vals=range_y))
    node_list.append(node_range_y)

    range_x_new_shape = [1, input_width]
    range_y_new_shape = [input_height, 1]

    name_range_x_new_shape = [model_name + "@range_x_new_shape"]
    node_range_x_new_shape = onnx.helper.make_node(
        'Constant',
        inputs=[],
        outputs=name_range_x_new_shape,
        value=onnx.helper.make_tensor(
            name=name_range_x_new_shape[0] + "@const",
            data_type=onnx.TensorProto.INT64,
            dims=[len(range_x_new_shape)],
            vals=range_x_new_shape))
    node_list.append(node_range_x_new_shape)

    name_range_y_new_shape = [model_name + "@range_y_new_shape"]
    node_range_y_new_shape = onnx.helper.make_node(
        'Constant',
        inputs=[],
        outputs=name_range_y_new_shape,
        value=onnx.helper.make_tensor(
            name=name_range_y_new_shape[0] + "@const",
            data_type=onnx.TensorProto.INT64,
            dims=[len(range_y_new_shape)],
            vals=range_y_new_shape))
    node_list.append(node_range_y_new_shape)

    outputs_range_x_reshape = [model_name + "@range_x_reshape"]
    node_range_x_reshape = onnx.helper.make_node(
        'Reshape',
        inputs=name_range_x + name_range_x_new_shape,
        outputs=outputs_range_x_reshape)
    node_list.append(node_range_x_reshape)

    outputs_range_y_reshape = [model_name + "@range_y_reshape"]
    node_range_y_reshape = onnx.helper.make_node(
        'Reshape',
        inputs=name_range_y + name_range_y_new_shape,
        outputs=outputs_range_y_reshape)
    node_list.append(node_range_y_reshape)

    outputs_grid_x = [model_name + "@grid_x"]
    node_grid_x = onnx.helper.make_node(
        "Tile",
        inputs=outputs_range_x_reshape + name_range_y_new_shape,
        outputs=outputs_grid_x)
    node_list.append(node_grid_x)

    outputs_grid_y = [model_name + "@grid_y"]
    node_grid_y = onnx.helper.make_node(
        "Tile",
        inputs=outputs_range_y_reshape + name_range_x_new_shape,
        outputs=outputs_grid_y)
    node_list.append(node_grid_y)

    outputs_box_x = [model_name + "@box_x"]
    outputs_box_y = [model_name + "@box_y"]
    outputs_box_w = [model_name + "@box_w"]
    outputs_box_h = [model_name + "@box_h"]
    outputs_conf = [model_name + "@conf"]
    outputs_prob = [model_name + "@prob"]

    node_split_input = onnx.helper.make_node(
        "Split",
        inputs=outputs_x_transpose,
        outputs=outputs_box_x + outputs_box_y + outputs_box_w\
                + outputs_box_h + outputs_conf + outputs_prob,
        axis=-1,
        split=[1, 1, 1, 1, 1, class_num])
    node_list.append(node_split_input)

    outputs_box_x_sigmoid = [model_name + "@box_x_sigmoid"]
    outputs_box_y_sigmoid = [model_name + "@box_y_sigmoid"]

    node_box_x_sigmoid = onnx.helper.make_node(
        "Sigmoid", inputs=outputs_box_x, outputs=outputs_box_x_sigmoid)
    node_list.append(node_box_x_sigmoid)

    node_box_y_sigmoid = onnx.helper.make_node(
        "Sigmoid", inputs=outputs_box_y, outputs=outputs_box_y_sigmoid)
    node_list.append(node_box_y_sigmoid)

    outputs_box_x_squeeze = [model_name + "@box_x_squeeze"]
    outputs_box_y_squeeze = [model_name + "@box_y_squeeze"]

    node_box_x_squeeze = onnx.helper.make_node(
        'Squeeze',
        inputs=outputs_box_x_sigmoid,
        outputs=outputs_box_x_squeeze,
        axes=[4])
    node_list.append(node_box_x_squeeze)

    node_box_y_squeeze = onnx.helper.make_node(
        'Squeeze',
        inputs=outputs_box_y_sigmoid,
        outputs=outputs_box_y_squeeze,
        axes=[4])
    node_list.append(node_box_y_squeeze)

    outputs_box_x_add_grid = [model_name + "@box_x_add_grid"]
    outputs_box_y_add_grid = [model_name + "@box_y_add_grid"]

    node_box_x_add_grid = onnx.helper.make_node(
        "Add",
        inputs=outputs_grid_x + outputs_box_x_squeeze,
        outputs=outputs_box_x_add_grid)
    node_list.append(node_box_x_add_grid)

    node_box_y_add_grid = onnx.helper.make_node(
        "Add",
        inputs=outputs_grid_y + outputs_box_y_squeeze,
        outputs=outputs_box_y_add_grid)
    node_list.append(node_box_y_add_grid)

    name_input_h = [model_name + "@input_h"]
    name_input_w = [model_name + "@input_w"]

    node_input_h = onnx.helper.make_node(
        'Constant',
        inputs=[],
        outputs=name_input_h,
        value=onnx.helper.make_tensor(
            name=name_input_w[0] + "@const",
            data_type=onnx.TensorProto.FLOAT,
            dims=(),
            vals=[input_height]))
    node_list.append(node_input_h)

    node_input_w = onnx.helper.make_node(
        'Constant',
        inputs=[],
        outputs=name_input_w,
        value=onnx.helper.make_tensor(
            name=name_input_w[0] + "@const",
            data_type=onnx.TensorProto.FLOAT,
            dims=(),
            vals=[input_width]))
    node_list.append(node_input_w)

    outputs_box_x_encode = [model_name + "@box_x_encode"]
    outputs_box_y_encode = [model_name + "@box_y_encode"]

    node_box_x_encode = onnx.helper.make_node(
        'Div',
        inputs=outputs_box_x_add_grid + name_input_w,
        outputs=outputs_box_x_encode)
    node_list.append(node_box_x_encode)

    node_box_y_encode = onnx.helper.make_node(
        'Div',
        inputs=outputs_box_y_add_grid + name_input_h,
        outputs=outputs_box_y_encode)
    node_list.append(node_box_y_encode)

    name_anchor_tensor = [model_name + "@anchor_tensor"]
    node_anchor_tensor = onnx.helper.make_node(
        "Constant",
        inputs=[],
        outputs=name_anchor_tensor,
        value=onnx.helper.make_tensor(
            name=name_anchor_tensor[0] + "@const",
            data_type=onnx.TensorProto.FLOAT,
            dims=[len(anchors)],
            vals=anchors))
    node_list.append(node_anchor_tensor)

    anchor_shape = [int(num_anchors), 2]
    name_anchor_shape = [model_name + "@anchor_shape"]
    node_anchor_shape = onnx.helper.make_node(
        "Constant",
        inputs=[],
        outputs=name_anchor_shape,
        value=onnx.helper.make_tensor(
            name=name_anchor_shape[0] + "@const",
            data_type=onnx.TensorProto.INT64,
            dims=[2],
            vals=anchor_shape))
    node_list.append(node_anchor_shape)

    outputs_anchor_tensor_reshape = [model_name + "@anchor_tensor_reshape"]
    node_anchor_tensor_reshape = onnx.helper.make_node(
        "Reshape",
        inputs=name_anchor_tensor + name_anchor_shape,
        outputs=outputs_anchor_tensor_reshape)
    node_list.append(node_anchor_tensor_reshape)

    name_input_size = [model_name + "@input_size"]
    node_input_size = onnx.helper.make_node(
        "Constant",
        inputs=[],
        outputs=name_input_size,
        value=onnx.helper.make_tensor(
            name=name_input_size[0] + "@const",
            data_type=onnx.TensorProto.FLOAT,
            dims=(),
            vals=[input_size]))
    node_list.append(node_input_size)

    outputs_anchors_div_input_size = [model_name + "@anchors_div_input_size"]
    node_anchors_div_input_size = onnx.helper.make_node(
        "Div",
        inputs=outputs_anchor_tensor_reshape + name_input_size,
        outputs=outputs_anchors_div_input_size)
    node_list.append(node_anchors_div_input_size)

    outputs_anchor_w = [model_name + "@anchor_w"]
    outputs_anchor_h = [model_name + "@anchor_h"]

    node_anchor_split = onnx.helper.make_node(
        'Split',
        inputs=outputs_anchors_div_input_size,
        outputs=outputs_anchor_w + outputs_anchor_h,
        axis=1,
        split=[1, 1])
    node_list.append(node_anchor_split)

    new_anchor_shape = [1, int(num_anchors), 1, 1]
    name_new_anchor_shape = [model_name + "@new_anchor_shape"]
    node_new_anchor_shape = onnx.helper.make_node(
        'Constant',
        inputs=[],
        outputs=name_new_anchor_shape,
        value=onnx.helper.make_tensor(
            name=name_new_anchor_shape[0] + "@const",
            data_type=onnx.TensorProto.INT64,
            dims=[len(new_anchor_shape)],
            vals=new_anchor_shape))
    node_list.append(node_new_anchor_shape)

    outputs_anchor_w_reshape = [model_name + "@anchor_w_reshape"]
    outputs_anchor_h_reshape = [model_name + "@anchor_h_reshape"]

    node_anchor_w_reshape = onnx.helper.make_node(
        'Reshape',
        inputs=outputs_anchor_w + name_new_anchor_shape,
        outputs=outputs_anchor_w_reshape)
    node_list.append(node_anchor_w_reshape)

    node_anchor_h_reshape = onnx.helper.make_node(
        'Reshape',
        inputs=outputs_anchor_h + name_new_anchor_shape,
        outputs=outputs_anchor_h_reshape)
    node_list.append(node_anchor_h_reshape)

    outputs_box_w_squeeze = [model_name + "@box_w_squeeze"]
    node_box_w_squeeze = onnx.helper.make_node(
        'Squeeze',
        inputs=outputs_box_w,
        outputs=outputs_box_w_squeeze,
        axes=[4])
    node_list.append(node_box_w_squeeze)

    outputs_box_h_squeeze = [model_name + "@box_h_squeeze"]
    node_box_h_squeeze = onnx.helper.make_node(
        'Squeeze',
        inputs=outputs_box_h,
        outputs=outputs_box_h_squeeze,
        axes=[4])
    node_list.append(node_box_h_squeeze)

    outputs_box_w_exp = [model_name + "@box_w_exp"]
    node_box_w_exp = onnx.helper.make_node(
        "Exp", inputs=outputs_box_w_squeeze, outputs=outputs_box_w_exp)
    node_list.append(node_box_w_exp)

    outputs_box_h_exp = [model_name + "@box_h_exp"]
    node_box_h_exp = onnx.helper.make_node(
        "Exp", inputs=outputs_box_h_squeeze, outputs=outputs_box_h_exp)
    node_list.append(node_box_h_exp)

    outputs_box_w_encode = [model_name + "box_w_encode"]
    outputs_box_h_encode = [model_name + "box_h_encode"]

    node_box_w_encode = onnx.helper.make_node(
        'Mul',
        inputs=outputs_box_w_exp + outputs_anchor_w_reshape,
        outputs=outputs_box_w_encode)
    node_list.append(node_box_w_encode)

    node_box_h_encode = onnx.helper.make_node(
        'Mul',
        inputs=outputs_box_h_exp + outputs_anchor_h_reshape,
        outputs=outputs_box_h_encode)
    node_list.append(node_box_h_encode)

    outputs_conf_sigmoid = [model_name + "@conf_sigmoid"]
    node_conf_sigmoid = onnx.helper.make_node(
        'Sigmoid', inputs=outputs_conf, outputs=outputs_conf_sigmoid)
    node_list.append(node_conf_sigmoid)

    name_conf_thresh = [model_name + "@conf_thresh"]
    node_conf_thresh = onnx.helper.make_node(
        'Constant',
        inputs=[],
        outputs=name_conf_thresh,
        value=onnx.helper.make_tensor(
            name=name_conf_thresh[0] + "@const",
            data_type=onnx.TensorProto.FLOAT,
            dims=[num_anchors * input_height * input_width],
            vals=conf_thresh_mat))
    node_list.append(node_conf_thresh)

    conf_shape = [1, int(num_anchors), input_height, input_width, 1]
    name_conf_shape = [model_name + "@conf_shape"]
    node_conf_shape = onnx.helper.make_node(
        'Constant',
        inputs=[],
        outputs=name_conf_shape,
        value=onnx.helper.make_tensor(
            name=name_conf_shape[0] + "@const",
            data_type=onnx.TensorProto.INT64,
            dims=[len(conf_shape)],
            vals=conf_shape))
    node_list.append(node_conf_shape)

    outputs_conf_thresh_reshape = [model_name + "@conf_thresh_reshape"]
    node_conf_thresh_reshape = onnx.helper.make_node(
        'Reshape',
        inputs=name_conf_thresh + name_conf_shape,
        outputs=outputs_conf_thresh_reshape)
    node_list.append(node_conf_thresh_reshape)

    outputs_conf_sub = [model_name + "@conf_sub"]
    node_conf_sub = onnx.helper.make_node(
        'Sub',
        inputs=outputs_conf_sigmoid + outputs_conf_thresh_reshape,
        outputs=outputs_conf_sub)
    node_list.append(node_conf_sub)

    outputs_conf_clip = [model_name + "@conf_clip"]
    node_conf_clip = onnx.helper.make_node(
        'Clip', inputs=outputs_conf_sub, outputs=outputs_conf_clip)
    node_list.append(node_conf_clip)

    zeros = [0]
    name_zeros = [model_name + "@zeros"]
    node_zeros = onnx.helper.make_node(
        'Constant',
        inputs=[],
        outputs=name_zeros,
        value=onnx.helper.make_tensor(
            name=name_zeros[0] + "@const",
            data_type=onnx.TensorProto.FLOAT,
            dims=(),
            vals=zeros))
    node_list.append(node_zeros)

    outputs_conf_clip_bool = [model_name + "@conf_clip_bool"]
    node_conf_clip_bool = onnx.helper.make_node(
        'Greater',
        inputs=outputs_conf_clip + name_zeros,
        outputs=outputs_conf_clip_bool)
    node_list.append(node_conf_clip_bool)

    outputs_conf_clip_cast = [model_name + "@conf_clip_cast"]
    node_conf_clip_cast = onnx.helper.make_node(
        'Cast',
        inputs=outputs_conf_clip_bool,
        outputs=outputs_conf_clip_cast,
        to=1)
    node_list.append(node_conf_clip_cast)

    outputs_conf_set_zero = [model_name + "@conf_set_zero"]
    node_conf_set_zero = onnx.helper.make_node(
        'Mul',
        inputs=outputs_conf_sigmoid + outputs_conf_clip_cast,
        outputs=outputs_conf_set_zero)
    node_list.append(node_conf_set_zero)

    outputs_prob_sigmoid = [model_name + "@prob_sigmoid"]
    node_prob_sigmoid = onnx.helper.make_node(
        'Sigmoid', inputs=outputs_prob, outputs=outputs_prob_sigmoid)
    node_list.append(node_prob_sigmoid)

    new_shape = [1, int(num_anchors), input_height, input_width, 1]
    name_new_shape = [model_name + "@new_shape"]
    node_new_shape = onnx.helper.make_node(
        'Constant',
        inputs=[],
        outputs=name_new_shape,
        value=onnx.helper.make_tensor(
            name=name_new_shape[0] + "@const",
            data_type=onnx.TensorProto.INT64,
            dims=[len(new_shape)],
            vals=new_shape))
    node_list.append(node_new_shape)

    outputs_conf_new_shape = [model_name + "@_conf_new_shape"]
    node_conf_new_shape = onnx.helper.make_node(
        'Reshape',
        inputs=outputs_conf_set_zero + name_new_shape,
        outputs=outputs_conf_new_shape)
    node_list.append(node_conf_new_shape)

    outputs_score = [model_name + "@score"]
    node_score = onnx.helper.make_node(
        'Mul',
        inputs=outputs_prob_sigmoid + outputs_conf_new_shape,
        outputs=outputs_score)
    node_list.append(node_score)

    outputs_conf_bool = [model_name + "@conf_bool"]
    node_conf_bool = onnx.helper.make_node(
        'Greater',
        inputs=outputs_conf_new_shape + name_zeros,
        outputs=outputs_conf_bool)
    node_list.append(node_conf_bool)

    outputs_box_x_new_shape = [model_name + "@box_x_new_shape"]
    node_box_x_new_shape = onnx.helper.make_node(
        'Reshape',
        inputs=outputs_box_x_encode + name_new_shape,
        outputs=outputs_box_x_new_shape)
    node_list.append(node_box_x_new_shape)

    outputs_box_y_new_shape = [model_name + "@box_y_new_shape"]
    node_box_y_new_shape = onnx.helper.make_node(
        'Reshape',
        inputs=outputs_box_y_encode + name_new_shape,
        outputs=outputs_box_y_new_shape)
    node_list.append(node_box_y_new_shape)

    outputs_box_w_new_shape = [model_name + "@box_w_new_shape"]
    node_box_w_new_shape = onnx.helper.make_node(
        'Reshape',
        inputs=outputs_box_w_encode + name_new_shape,
        outputs=outputs_box_w_new_shape)
    node_list.append(node_box_w_new_shape)

    outputs_box_h_new_shape = [model_name + "@box_h_new_shape"]
    node_box_h_new_shape = onnx.helper.make_node(
        'Reshape',
        inputs=outputs_box_h_encode + name_new_shape,
        outputs=outputs_box_h_new_shape)
    node_list.append(node_box_h_new_shape)

    outputs_pred_box = [model_name + "@pred_box"]
    node_pred_box = onnx.helper.make_node(
        'Concat',
        inputs=outputs_box_x_new_shape + outputs_box_y_new_shape + \
               outputs_box_w_new_shape + outputs_box_h_new_shape,
        outputs=outputs_pred_box,
        axis=4)
    node_list.append(node_pred_box)

    outputs_conf_cast = [model_name + "conf_cast"]
    node_conf_cast = onnx.helper.make_node(
        'Cast', inputs=outputs_conf_bool, outputs=outputs_conf_cast, to=1)
    node_list.append(node_conf_cast)

    outputs_pred_box_mul_conf = [model_name + "@pred_box_mul_conf"]
    node_pred_box_mul_conf = onnx.helper.make_node(
        'Mul',
        inputs=outputs_pred_box + outputs_conf_cast,
        outputs=outputs_pred_box_mul_conf)
    node_list.append(node_pred_box_mul_conf)

    box_shape = [1, int(num_anchors) * input_height * input_width, 4]
    name_box_shape = [model_name + "@box_shape"]
    node_box_shape = onnx.helper.make_node(
        'Constant',
        inputs=[],
        outputs=name_box_shape,
        value=onnx.helper.make_tensor(
            name=name_box_shape[0] + "@const",
            data_type=onnx.TensorProto.INT64,
            dims=[len(box_shape)],
            vals=box_shape))
    node_list.append(node_box_shape)

    outputs_pred_box_new_shape = [model_name + "@pred_box_new_shape"]
    node_pred_box_new_shape = onnx.helper.make_node(
        'Reshape',
        inputs=outputs_pred_box_mul_conf + name_box_shape,
        outputs=outputs_pred_box_new_shape)
    node_list.append(node_pred_box_new_shape)

    outputs_pred_box_x = [model_name + "@_pred_box_x"]
    outputs_pred_box_y = [model_name + "@_pred_box_y"]
    outputs_pred_box_w = [model_name + "@_pred_box_w"]
    outputs_pred_box_h = [model_name + "@_pred_box_h"]

    node_pred_box_split = onnx.helper.make_node(
        'Split',
        inputs=outputs_pred_box_new_shape,
        outputs=outputs_pred_box_x + outputs_pred_box_y + outputs_pred_box_w +
        outputs_pred_box_h,
        axis=2)
    node_list.append(node_pred_box_split)

    name_number_2 = [model_name + "@number_2"]
    node_number_2 = onnx.helper.make_node(
        "Constant",
        inputs=[],
        outputs=name_number_2,
        value=onnx.helper.make_tensor(
            name=name_number_2[0] + "@const",
            data_type=onnx.TensorProto.FLOAT,
            dims=(),
            vals=[2]))
    node_list.append(node_number_2)

    outputs_half_w = [model_name + "@half_w"]
    node_half_w = onnx.helper.make_node(
        "Div",
        inputs=outputs_pred_box_w + name_number_2,
        outputs=outputs_half_w)
    node_list.append(node_half_w)

    outputs_half_h = [model_name + "@half_h"]
    node_half_h = onnx.helper.make_node(
        "Div",
        inputs=outputs_pred_box_h + name_number_2,
        outputs=outputs_half_h)
    node_list.append(node_half_h)

    outputs_pred_box_x1 = [model_name + "@pred_box_x1"]
    node_pred_box_x1 = onnx.helper.make_node(
        'Sub',
        inputs=outputs_pred_box_x + outputs_half_w,
        outputs=outputs_pred_box_x1)
    node_list.append(node_pred_box_x1)

    outputs_pred_box_y1 = [model_name + "@pred_box_y1"]
    node_pred_box_y1 = onnx.helper.make_node(
        'Sub',
        inputs=outputs_pred_box_y + outputs_half_h,
        outputs=outputs_pred_box_y1)
    node_list.append(node_pred_box_y1)

    outputs_pred_box_x2 = [model_name + "@pred_box_x2"]
    node_pred_box_x2 = onnx.helper.make_node(
        'Add',
        inputs=outputs_pred_box_x + outputs_half_w,
        outputs=outputs_pred_box_x2)
    node_list.append(node_pred_box_x2)

    outputs_pred_box_y2 = [model_name + "@pred_box_y2"]
    node_pred_box_y2 = onnx.helper.make_node(
        'Add',
        inputs=outputs_pred_box_y + outputs_half_h,
        outputs=outputs_pred_box_y2)
    node_list.append(node_pred_box_y2)

    outputs_sqeeze_image_size = [model_name + "@sqeeze_image_size"]
    node_sqeeze_image_size = onnx.helper.make_node(
        "Squeeze",
        axes=[0],
        inputs=image_size,
        outputs=outputs_sqeeze_image_size)
    node_list.append(node_sqeeze_image_size)

    output_img_height = [model_name + "@img_height"]
    output_img_width = [model_name + "@img_width"]
    node_image_size_split = onnx.helper.make_node(
        "Split",
        inputs=outputs_sqeeze_image_size,
        outputs=output_img_height + output_img_width,
        axis=-1,
        split=[1, 1])
    node_list.append(node_image_size_split)

    output_img_width_cast = [model_name + "@img_width_cast"]
    node_img_width_cast = onnx.helper.make_node(
        'Cast', inputs=output_img_width, outputs=output_img_width_cast, to=1)
    node_list.append(node_img_width_cast)

    output_img_height_cast = [model_name + "@img_height_cast"]
    node_img_height_cast = onnx.helper.make_node(
        'Cast', inputs=output_img_height, outputs=output_img_height_cast, to=1)
    node_list.append(node_img_height_cast)

    outputs_pred_box_x1_decode = [model_name + "@pred_box_x1_decode"]
    outputs_pred_box_y1_decode = [model_name + "@pred_box_y1_decode"]
    outputs_pred_box_x2_decode = [model_name + "@pred_box_x2_decode"]
    outputs_pred_box_y2_decode = [model_name + "@pred_box_y2_decode"]

    node_pred_box_x1_decode = onnx.helper.make_node(
        'Mul',
        inputs=outputs_pred_box_x1 + output_img_width_cast,
        outputs=outputs_pred_box_x1_decode)
    node_list.append(node_pred_box_x1_decode)

    node_pred_box_y1_decode = onnx.helper.make_node(
        'Mul',
        inputs=outputs_pred_box_y1 + output_img_height_cast,
        outputs=outputs_pred_box_y1_decode)
    node_list.append(node_pred_box_y1_decode)

    node_pred_box_x2_decode = onnx.helper.make_node(
        'Mul',
        inputs=outputs_pred_box_x2 + output_img_width_cast,
        outputs=outputs_pred_box_x2_decode)
    node_list.append(node_pred_box_x2_decode)

    node_pred_box_y2_decode = onnx.helper.make_node(
        'Mul',
        inputs=outputs_pred_box_y2 + output_img_height_cast,
        outputs=outputs_pred_box_y2_decode)
    node_list.append(node_pred_box_y2_decode)

    name_number_one = [model_name + "@one"]
    node_number_one = onnx.helper.make_node(
        'Constant',
        inputs=[],
        outputs=name_number_one,
        value=onnx.helper.make_tensor(
            name=name_number_one[0] + "@const",
            data_type=onnx.TensorProto.FLOAT,
            dims=(),
            vals=[1]))
    node_list.append(node_number_one)

    output_new_img_height = [model_name + "@new_img_height"]
    node_new_img_height = onnx.helper.make_node(
        'Sub',
        inputs=output_img_height_cast + name_number_one,
        outputs=output_new_img_height)
    node_list.append(node_new_img_height)

    output_new_img_width = [model_name + "@new_img_width"]
    node_new_img_width = onnx.helper.make_node(
        'Sub',
        inputs=output_img_width_cast + name_number_one,
        outputs=output_new_img_width)
    node_list.append(node_new_img_width)

    outputs_pred_box_x2_sub_w = [model_name + "@pred_box_x2_sub_w"]
    node_pred_box_x2_sub_w = onnx.helper.make_node(
        'Sub',
        inputs=outputs_pred_box_x2_decode + output_new_img_width,
        outputs=outputs_pred_box_x2_sub_w)
    node_list.append(node_pred_box_x2_sub_w)

    outputs_pred_box_y2_sub_h = [model_name + "@pred_box_y2_sub_h"]
    node_pred_box_y2_sub_h = onnx.helper.make_node(
        'Sub',
        inputs=outputs_pred_box_y2_decode + output_new_img_height,
        outputs=outputs_pred_box_y2_sub_h)
    node_list.append(node_pred_box_y2_sub_h)

    outputs_pred_box_x1_clip = [model_name + "@pred_box_x1_clip"]
    outputs_pred_box_y1_clip = [model_name + "@pred_box_y1_clip"]
    outputs_pred_box_x2_clip = [model_name + "@pred_box_x2_clip"]
    outputs_pred_box_y2_clip = [model_name + "@pred_box_y2_clip"]

    node_pred_box_x1_clip = onnx.helper.make_node(
        'Clip',
        inputs=outputs_pred_box_x1_decode,
        outputs=outputs_pred_box_x1_clip,
        min=0.0,
        max=float(np.inf))
    node_list.append(node_pred_box_x1_clip)

    node_pred_box_y1_clip = onnx.helper.make_node(
        'Clip',
        inputs=outputs_pred_box_y1_decode,
        outputs=outputs_pred_box_y1_clip,
        min=0.0,
        max=float(np.inf))
    node_list.append(node_pred_box_y1_clip)

    node_pred_box_x2_clip = onnx.helper.make_node(
        'Clip',
        inputs=outputs_pred_box_x2_sub_w,
        outputs=outputs_pred_box_x2_clip,
        min=0.0,
        max=float(np.inf))
    node_list.append(node_pred_box_x2_clip)

    node_pred_box_y2_clip = onnx.helper.make_node(
        'Clip',
        inputs=outputs_pred_box_y2_sub_h,
        outputs=outputs_pred_box_y2_clip,
        min=0.0,
        max=float(np.inf))
    node_list.append(node_pred_box_y2_clip)

    outputs_pred_box_x2_res = [model_name + "@box_x2_res"]
    node_pred_box_x2_res = onnx.helper.make_node(
        'Sub',
        inputs=outputs_pred_box_x2_decode + outputs_pred_box_x2_clip,
        outputs=outputs_pred_box_x2_res)
    node_list.append(node_pred_box_x2_res)

    outputs_pred_box_y2_res = [model_name + "@box_y2_res"]
    node_pred_box_y2_res = onnx.helper.make_node(
        'Sub',
        inputs=outputs_pred_box_y2_decode + outputs_pred_box_y2_clip,
        outputs=outputs_pred_box_y2_res)
    node_list.append(node_pred_box_y2_res)

    node_pred_box_result = onnx.helper.make_node(
        'Concat',
        inputs=outputs_pred_box_x1_clip + outputs_pred_box_y1_clip +
        outputs_pred_box_x2_res + outputs_pred_box_y2_res,
        outputs=outputs['Boxes'],
        axis=-1)
    node_list.append(node_pred_box_result)

    score_shape = [1, input_height * input_width * int(num_anchors), class_num]
    name_score_shape = [model_name + "@score_shape"]
    node_score_shape = onnx.helper.make_node(
        "Constant",
        inputs=[],
        outputs=name_score_shape,
        value=onnx.helper.make_tensor(
            name=name_score_shape[0] + "@const",
            data_type=onnx.TensorProto.INT64,
            dims=[len(score_shape)],
            vals=score_shape))
    node_list.append(node_score_shape)

    node_score_new_shape = onnx.helper.make_node(
        'Reshape',
        inputs=outputs_score + name_score_shape,
        outputs=outputs['Scores'])
    node_list.append(node_score_new_shape)
    return tuple(node_list)


def multiclass_nms_op(operator, block):
    """
    Convert the paddle multiclass_nms to onnx op.
    This op is get the select boxes from origin boxes.
    """
    inputs, attrs, outputs = op_io_info(operator)
    result_name = outputs['Out'][0]
    background = attrs['background_label']
    normalized = attrs['normalized']
    if normalized == False:
        warnings.warn(
            'The parameter normalized of multiclass_nms OP of Paddle is False, which has diff with ONNX. \
                         Please set normalized=True in multiclass_nms of Paddle')

    #convert the paddle attribute to onnx tensor 
    name_score_threshold = [outputs['Out'][0] + "@score_threshold"]
    name_iou_threshold = [outputs['Out'][0] + "@iou_threshold"]
    name_keep_top_k = [outputs['Out'][0] + '@keep_top_k']
    name_keep_top_k_2D = [outputs['Out'][0] + '@keep_top_k_1D']

    node_score_threshold = onnx.helper.make_node(
        'Constant',
        inputs=[],
        outputs=name_score_threshold,
        value=onnx.helper.make_tensor(
            name=name_score_threshold[0] + "@const",
            data_type=onnx.TensorProto.FLOAT,
            dims=(),
            vals=[float(attrs['score_threshold'])]))

    node_iou_threshold = onnx.helper.make_node(
        'Constant',
        inputs=[],
        outputs=name_iou_threshold,
        value=onnx.helper.make_tensor(
            name=name_iou_threshold[0] + "@const",
            data_type=onnx.TensorProto.FLOAT,
            dims=(),
            vals=[float(attrs['nms_threshold'])]))

    node_keep_top_k = onnx.helper.make_node(
        'Constant',
        inputs=[],
        outputs=name_keep_top_k,
        value=onnx.helper.make_tensor(
            name=name_keep_top_k[0] + "@const",
            data_type=onnx.TensorProto.INT64,
            dims=(),
            vals=[np.int64(attrs['keep_top_k'])]))

    node_keep_top_k_2D = onnx.helper.make_node(
        'Constant',
        inputs=[],
        outputs=name_keep_top_k_2D,
        value=onnx.helper.make_tensor(
            name=name_keep_top_k_2D[0] + "@const",
            data_type=onnx.TensorProto.INT64,
            dims=[1, 1],
            vals=[np.int64(attrs['keep_top_k'])]))

    # the paddle data format is x1,y1,x2,y2
    kwargs = {'center_point_box': 0}

    name_select_nms = [outputs['Out'][0] + "@select_index"]
    node_select_nms= onnx.helper.make_node(
        'NonMaxSuppression',
        inputs=inputs['BBoxes'] + inputs['Scores'] + name_keep_top_k +\
            name_iou_threshold + name_score_threshold,
        outputs=name_select_nms)
    # step 1 nodes select the nms class 
    node_list = [
        node_score_threshold, node_iou_threshold, node_keep_top_k,
        node_keep_top_k_2D, node_select_nms
    ]

    # create some const value to use 
    name_const_value = [result_name+"@const_0",
        result_name+"@const_1",\
        result_name+"@const_2",\
        result_name+"@const_-1"]
    value_const_value = [0, 1, 2, -1]
    for name, value in zip(name_const_value, value_const_value):
        node = onnx.helper.make_node(
            'Constant',
            inputs=[],
            outputs=[name],
            value=onnx.helper.make_tensor(
                name=name + "@const",
                data_type=onnx.TensorProto.INT64,
                dims=[1],
                vals=[value]))
        node_list.append(node)

    # Ine this code block, we will deocde the raw score data, reshape N * C * M to 1 * N*C*M 
    # and the same time, decode the select indices to 1 * D, gather the select_indices
    outputs_gather_1 = [result_name + "@gather_1"]
    node_gather_1 = onnx.helper.make_node(
        'Gather',
        inputs=name_select_nms + [result_name + "@const_1"],
        outputs=outputs_gather_1,
        axis=1)
    node_list.append(node_gather_1)

    outputs_squeeze_gather_1 = [result_name + "@sequeeze_gather_1"]
    node_squeeze_gather_1 = onnx.helper.make_node(
        'Squeeze',
        inputs=outputs_gather_1,
        outputs=outputs_squeeze_gather_1,
        axes=[1])
    node_list.append(node_squeeze_gather_1)

    outputs_gather_2 = [result_name + "@gather_2"]
    node_gather_2 = onnx.helper.make_node(
        'Gather',
        inputs=name_select_nms + [result_name + "@const_2"],
        outputs=outputs_gather_2,
        axis=1)
    node_list.append(node_gather_2)

    #slice the class is not 0 
    if background == 0:
        outputs_nonzero = [result_name + "@nonzero"]
        node_nonzero = onnx.helper.make_node(
            'NonZero', inputs=outputs_squeeze_gather_1, outputs=outputs_nonzero)
        node_list.append(node_nonzero)
    else:
        name_thresh = [result_name + "@thresh"]
        node_thresh = onnx.helper.make_node(
            'Constant',
            inputs=[],
            outputs=name_thresh,
            value=onnx.helper.make_tensor(
                name=name_thresh[0] + "@const",
                data_type=onnx.TensorProto.INT32,
                dims=[1],
                vals=[-1]))
        node_list.append(node_thresh)

        outputs_cast = [result_name + "@cast"]
        node_cast = onnx.helper.make_node(
            'Cast', inputs=outputs_squeeze_gather_1, outputs=outputs_cast, to=6)
        node_list.append(node_cast)

        outputs_greater = [result_name + "@greater"]
        node_greater = onnx.helper.make_node(
            'Greater',
            inputs=outputs_cast + name_thresh,
            outputs=outputs_greater)
        node_list.append(node_greater)

        outputs_nonzero = [result_name + "@nonzero"]
        node_nonzero = onnx.helper.make_node(
            'NonZero', inputs=outputs_greater, outputs=outputs_nonzero)
        node_list.append(node_nonzero)

    outputs_gather_1_nonzero = [result_name + "@gather_1_nonzero"]
    node_gather_1_nonzero = onnx.helper.make_node(
        'Gather',
        inputs=outputs_gather_1 + outputs_nonzero,
        outputs=outputs_gather_1_nonzero,
        axis=0)
    node_list.append(node_gather_1_nonzero)

    outputs_gather_2_nonzero = [result_name + "@gather_2_nonzero"]
    node_gather_2_nonzero = onnx.helper.make_node(
        'Gather',
        inputs=outputs_gather_2 + outputs_nonzero,
        outputs=outputs_gather_2_nonzero,
        axis=0)
    node_list.append(node_gather_2_nonzero)

    # reshape scores N * C * M to (N*C*M) * 1 
    outputs_reshape_scores_rank1 = [result_name + "@reshape_scores_rank1"]
    node_reshape_scores_rank1 = onnx.helper.make_node(
        "Reshape",
        inputs=inputs['Scores'] + [result_name + "@const_-1"],
        outputs=outputs_reshape_scores_rank1)
    node_list.append(node_reshape_scores_rank1)

    # get the shape of scores 
    outputs_shape_scores = [result_name + "@shape_scores"]
    node_shape_scores = onnx.helper.make_node(
        'Shape', inputs=inputs['Scores'], outputs=outputs_shape_scores)
    node_list.append(node_shape_scores)

    # gather the index: 2 shape of scores 
    outputs_gather_scores_dim1 = [result_name + "@gather_scores_dim1"]
    node_gather_scores_dim1 = onnx.helper.make_node(
        'Gather',
        inputs=outputs_shape_scores + [result_name + "@const_2"],
        outputs=outputs_gather_scores_dim1,
        axis=0)
    node_list.append(node_gather_scores_dim1)

    # mul class * M 
    outputs_mul_classnum_boxnum = [result_name + "@mul_classnum_boxnum"]
    node_mul_classnum_boxnum = onnx.helper.make_node(
        'Mul',
        inputs=outputs_gather_1_nonzero + outputs_gather_scores_dim1,
        outputs=outputs_mul_classnum_boxnum)
    node_list.append(node_mul_classnum_boxnum)

    # add class * M * index 
    outputs_add_class_M_index = [result_name + "@add_class_M_index"]
    node_add_class_M_index = onnx.helper.make_node(
        'Add',
        inputs=outputs_mul_classnum_boxnum + outputs_gather_2_nonzero,
        outputs=outputs_add_class_M_index)
    node_list.append(node_add_class_M_index)

    # Squeeze the indices to 1 dim  
    outputs_squeeze_select_index = [result_name + "@squeeze_select_index"]
    node_squeeze_select_index = onnx.helper.make_node(
        'Squeeze',
        inputs=outputs_add_class_M_index,
        outputs=outputs_squeeze_select_index,
        axes=[0, 2])
    node_list.append(node_squeeze_select_index)

    # gather the data from flatten scores 
    outputs_gather_select_scores = [result_name + "@gather_select_scores"]
    node_gather_select_scores = onnx.helper.make_node('Gather',
        inputs=outputs_reshape_scores_rank1 + \
            outputs_squeeze_select_index,
        outputs=outputs_gather_select_scores,
        axis=0)
    node_list.append(node_gather_select_scores)

    # get nums to input TopK 
    outputs_shape_select_num = [result_name + "@shape_select_num"]
    node_shape_select_num = onnx.helper.make_node(
        'Shape',
        inputs=outputs_gather_select_scores,
        outputs=outputs_shape_select_num)
    node_list.append(node_shape_select_num)

    outputs_gather_select_num = [result_name + "@gather_select_num"]
    node_gather_select_num = onnx.helper.make_node(
        'Gather',
        inputs=outputs_shape_select_num + [result_name + "@const_0"],
        outputs=outputs_gather_select_num,
        axis=0)
    node_list.append(node_gather_select_num)

    outputs_unsqueeze_select_num = [result_name + "@unsqueeze_select_num"]
    node_unsqueeze_select_num = onnx.helper.make_node(
        'Unsqueeze',
        inputs=outputs_gather_select_num,
        outputs=outputs_unsqueeze_select_num,
        axes=[0])
    node_list.append(node_unsqueeze_select_num)

    outputs_concat_topK_select_num = [result_name + "@conat_topK_select_num"]
    node_conat_topK_select_num = onnx.helper.make_node(
        'Concat',
        inputs=outputs_unsqueeze_select_num + name_keep_top_k_2D,
        outputs=outputs_concat_topK_select_num,
        axis=0)
    node_list.append(node_conat_topK_select_num)

    outputs_cast_concat_topK_select_num = [
        result_name + "@concat_topK_select_num"
    ]
    node_outputs_cast_concat_topK_select_num = onnx.helper.make_node(
        'Cast',
        inputs=outputs_concat_topK_select_num,
        outputs=outputs_cast_concat_topK_select_num,
        to=6)
    node_list.append(node_outputs_cast_concat_topK_select_num)
    # get min(topK, num_select)
    outputs_compare_topk_num_select = [result_name + "@compare_topk_num_select"]
    node_compare_topk_num_select = onnx.helper.make_node(
        'ReduceMin',
        inputs=outputs_cast_concat_topK_select_num,
        outputs=outputs_compare_topk_num_select,
        keepdims=0)
    node_list.append(node_compare_topk_num_select)

    # unsqueeze the indices to 1D tensor 
    outputs_unsqueeze_topk_select_indices = [
        result_name + "@unsqueeze_topk_select_indices"
    ]
    node_unsqueeze_topk_select_indices = onnx.helper.make_node(
        'Unsqueeze',
        inputs=outputs_compare_topk_num_select,
        outputs=outputs_unsqueeze_topk_select_indices,
        axes=[0])
    node_list.append(node_unsqueeze_topk_select_indices)

    # cast the indices to INT64
    outputs_cast_topk_indices = [result_name + "@cast_topk_indices"]
    node_cast_topk_indices = onnx.helper.make_node(
        'Cast',
        inputs=outputs_unsqueeze_topk_select_indices,
        outputs=outputs_cast_topk_indices,
        to=7)
    node_list.append(node_cast_topk_indices)

    # select topk scores  indices
    outputs_topk_select_topk_indices = [result_name + "@topk_select_topk_values",\
        result_name + "@topk_select_topk_indices"]
    node_topk_select_topk_indices = onnx.helper.make_node(
        'TopK',
        inputs=outputs_gather_select_scores + outputs_cast_topk_indices,
        outputs=outputs_topk_select_topk_indices)
    node_list.append(node_topk_select_topk_indices)

    # gather topk label, scores, boxes
    outputs_gather_topk_scores = [result_name + "@gather_topk_scores"]
    node_gather_topk_scores = onnx.helper.make_node(
        'Gather',
        inputs=outputs_gather_select_scores +
        [outputs_topk_select_topk_indices[1]],
        outputs=outputs_gather_topk_scores,
        axis=0)
    node_list.append(node_gather_topk_scores)

    outputs_gather_topk_class = [result_name + "@gather_topk_class"]
    node_gather_topk_class = onnx.helper.make_node(
        'Gather',
        inputs=outputs_gather_1_nonzero +
        [outputs_topk_select_topk_indices[1]],
        outputs=outputs_gather_topk_class,
        axis=1)
    node_list.append(node_gather_topk_class)

    # gather the boxes need to gather the boxes id, then get boxes 
    outputs_gather_topk_boxes_id = [result_name + "@gather_topk_boxes_id"]
    node_gather_topk_boxes_id = onnx.helper.make_node(
        'Gather',
        inputs=outputs_gather_2_nonzero +
        [outputs_topk_select_topk_indices[1]],
        outputs=outputs_gather_topk_boxes_id,
        axis=1)
    node_list.append(node_gather_topk_boxes_id)

    # squeeze the gather_topk_boxes_id to 1 dim 
    outputs_squeeze_topk_boxes_id = [result_name + "@squeeze_topk_boxes_id"]
    node_squeeze_topk_boxes_id = onnx.helper.make_node(
        'Squeeze',
        inputs=outputs_gather_topk_boxes_id,
        outputs=outputs_squeeze_topk_boxes_id,
        axes=[0, 2])
    node_list.append(node_squeeze_topk_boxes_id)

    outputs_gather_select_boxes = [result_name + "@gather_select_boxes"]
    node_gather_select_boxes = onnx.helper.make_node(
        'Gather',
        inputs=inputs['BBoxes'] + outputs_squeeze_topk_boxes_id,
        outputs=outputs_gather_select_boxes,
        axis=1)
    node_list.append(node_gather_select_boxes)

    # concat the final result 
    # before concat need to cast the class to float 
    outputs_cast_topk_class = [result_name + "@cast_topk_class"]
    node_cast_topk_class = onnx.helper.make_node(
        'Cast',
        inputs=outputs_gather_topk_class,
        outputs=outputs_cast_topk_class,
        to=1)
    node_list.append(node_cast_topk_class)

    outputs_unsqueeze_topk_scores = [result_name + "@unsqueeze_topk_scores"]
    node_unsqueeze_topk_scores = onnx.helper.make_node(
        'Unsqueeze',
        inputs=outputs_gather_topk_scores,
        outputs=outputs_unsqueeze_topk_scores,
        axes=[0, 2])
    node_list.append(node_unsqueeze_topk_scores)

    inputs_concat_final_results = outputs_cast_topk_class + outputs_unsqueeze_topk_scores +\
        outputs_gather_select_boxes
    outputs_concat_final_results = outputs['Out']
    node_concat_final_results = onnx.helper.make_node(
        'Concat',
        inputs=inputs_concat_final_results,
        outputs=outputs_concat_final_results,
        axis=2)
    node_list.append(node_concat_final_results)

    return tuple(node_list)


def ExpandAspectRations(input_aspect_ratior, flip):
    expsilon = 1e-6
    output_ratios = [1.0]
    for input_ratio in input_aspect_ratior:
        already_exis = False
        for output_ratio in output_ratios:
            if abs(input_ratio - output_ratio) < expsilon:
                already_exis = True
                break
        if already_exis == False:
            output_ratios.append(input_ratio)
            if flip:
                output_ratios.append(1.0 / input_ratio)
    return output_ratios


def prior_box_op(operator, block):
    """
    In this function, use the attribute to get the prior box, because we do not use 
    the image data and feature map, wo could the python code to create the varaible, 
    and to create the onnx tensor as output.
    """
    inputs, attrs, outputs = op_io_info(operator)
    flip = bool(attrs['flip'])
    clip = bool(attrs['clip'])
    min_max_aspect_ratios_order = bool(attrs['min_max_aspect_ratios_order'])
    min_sizes = [float(size) for size in attrs['min_sizes']]
    max_sizes = [float(size) for size in attrs['max_sizes']]
    if isinstance(attrs['aspect_ratios'], list):
        aspect_ratios = [float(ratio) for ratio in attrs['aspect_ratios']]
    else:
        aspect_ratios = [float(attrs['aspect_ratios'])]
    variances = [float(var) for var in attrs['variances']]
    # set min_max_aspect_ratios_order = false 
    output_ratios = ExpandAspectRations(aspect_ratios, flip)

    step_w = float(attrs['step_w'])
    step_h = float(attrs['step_h'])
    offset = float(attrs['offset'])

    input_shape = block.vars[get_old_name(inputs['Input'][0])].shape
    image_shape = block.vars[get_old_name(inputs['Image'][0])].shape

    img_width = image_shape[3]
    img_height = image_shape[2]
    feature_width = input_shape[3]
    feature_height = input_shape[2]

    step_width = 1.0
    step_height = 1.0

    if step_w == 0.0 or step_h == 0.0:
        step_w = float(img_width / feature_width)
        step_h = float(img_height / feature_height)

    num_priors = len(output_ratios) * len(min_sizes)
    if len(max_sizes) > 0:
        num_priors += len(max_sizes)
    out_dim = (feature_height, feature_width, num_priors, 4)
    out_boxes = np.zeros(out_dim).astype('float32')
    out_var = np.zeros(out_dim).astype('float32')

    idx = 0
    for h in range(feature_height):
        for w in range(feature_width):
            c_x = (w + offset) * step_w
            c_y = (h + offset) * step_h
            idx = 0
            for s in range(len(min_sizes)):
                min_size = min_sizes[s]
                if not min_max_aspect_ratios_order:
                    # rest of priors
                    for r in range(len(output_ratios)):
                        ar = output_ratios[r]
                        c_w = min_size * math.sqrt(ar) / 2
                        c_h = (min_size / math.sqrt(ar)) / 2
                        out_boxes[h, w, idx, :] = [
                            (c_x - c_w) / img_width, (c_y - c_h) / img_height,
                            (c_x + c_w) / img_width, (c_y + c_h) / img_height
                        ]
                        idx += 1

                    if len(max_sizes) > 0:
                        max_size = max_sizes[s]
                        # second prior: aspect_ratio = 1,
                        c_w = c_h = math.sqrt(min_size * max_size) / 2
                        out_boxes[h, w, idx, :] = [
                            (c_x - c_w) / img_width, (c_y - c_h) / img_height,
                            (c_x + c_w) / img_width, (c_y + c_h) / img_height
                        ]
                        idx += 1
                else:
                    c_w = c_h = min_size / 2.
                    out_boxes[h, w, idx, :] = [
                        (c_x - c_w) / img_width, (c_y - c_h) / img_height,
                        (c_x + c_w) / img_width, (c_y + c_h) / img_height
                    ]
                    idx += 1
                    if len(max_sizes) > 0:
                        max_size = max_sizes[s]
                        # second prior: aspect_ratio = 1,
                        c_w = c_h = math.sqrt(min_size * max_size) / 2
                        out_boxes[h, w, idx, :] = [
                            (c_x - c_w) / img_width, (c_y - c_h) / img_height,
                            (c_x + c_w) / img_width, (c_y + c_h) / img_height
                        ]
                        idx += 1

                    # rest of priors
                    for r in range(len(output_ratios)):
                        ar = output_ratios[r]
                        if abs(ar - 1.) < 1e-6:
                            continue
                        c_w = min_size * math.sqrt(ar) / 2
                        c_h = (min_size / math.sqrt(ar)) / 2
                        out_boxes[h, w, idx, :] = [
                            (c_x - c_w) / img_width, (c_y - c_h) / img_height,
                            (c_x + c_w) / img_width, (c_y + c_h) / img_height
                        ]
                        idx += 1

    if clip:
        out_boxes = np.clip(out_boxes, 0.0, 1.0)
    # set the variance.
    out_var = np.tile(variances, (feature_height, feature_width, num_priors, 1))

    #make node that 
    node_boxes = onnx.helper.make_node(
        'Constant',
        inputs=[],
        outputs=outputs['Boxes'],
        value=onnx.helper.make_tensor(
            name=outputs['Boxes'][0] + "@const",
            data_type=onnx.TensorProto.FLOAT,
            dims=out_boxes.shape,
            vals=out_boxes.flatten()))
    node_vars = onnx.helper.make_node(
        'Constant',
        inputs=[],
        outputs=outputs['Variances'],
        value=onnx.helper.make_tensor(
            name=outputs['Variances'][0] + "@const",
            data_type=onnx.TensorProto.FLOAT,
            dims=out_var.shape,
            vals=out_var.flatten()))
    return (node_boxes, node_vars)


def box_coder_op(operator, block):
    """
   In this function, we will use the decode the prior box to target box,
   we just use the decode mode to transform this op.
   """
    inputs, attrs, outputs = op_io_info(operator)
    node_list = []

    prior_var = block.vars[get_old_name(inputs['PriorBox'][0])]
    t_size = block.vars[get_old_name(inputs['TargetBox'][0])].shape
    p_size = prior_var.shape

    # get the outout_name 
    result_name = outputs['OutputBox'][0]
    # n is size of batch, m is boxes num of targe_boxes
    n = t_size[0]
    m = t_size[0]

    axis = int(attrs['axis'])

    #norm
    norm = bool(attrs['box_normalized'])

    name_slice_x1 = outputs['OutputBox'][0] + "@x1"
    name_slice_y1 = outputs['OutputBox'][0] + "@y1"
    name_slice_x2 = outputs['OutputBox'][0] + "@x2"
    name_slice_y2 = outputs['OutputBox'][0] + "@y2"

    #make onnx tensor to save the intermeidate reslut 
    name_slice_indices = [[outputs['OutputBox'][0] + "@slice_" + str(i)]
                          for i in range(1, 3)]
    node_slice_indices = [None for i in range(1, 3)]

    # create the range(0, 4) const data to slice 
    for i in range(1, 3):
        node = onnx.helper.make_node(
            'Constant',
            inputs=[],
            outputs=name_slice_indices[i - 1],
            value=onnx.helper.make_tensor(
                name=name_slice_indices[i - 1][0] + "@const",
                data_type=onnx.TensorProto.FLOAT,
                dims=(),
                vals=[i]))
        node_list.append(node)
    # make node split data 
    name_box_split = [
        name_slice_x1, name_slice_y1, name_slice_x2, name_slice_y2
    ]
    split_shape = list(p_size)
    split_shape[-1] = 1

    node_split_prior_node = onnx.helper.make_node(
        'Split', inputs=inputs['PriorBox'], outputs=name_box_split, axis=1)
    node_list.append(node_split_prior_node)

    # make node get centor node for decode
    final_outputs_vars = []
    if not norm:
        name_centor_w_tmp = [outputs['OutputBox'][0] + "@centor_w_tmp"]
        name_centor_h_tmp = [outputs['OutputBox'][0] + "@centor_h_tmp"]
        node_centor_w_tmp = None
        node_centor_h_tmp = None
        name_centor_tmp_list = [name_centor_w_tmp, name_centor_h_tmp]
        node_centor_tmp_list = [node_centor_w_tmp, node_centor_h_tmp]

        count = 2
        for (name, node) in zip(name_centor_tmp_list, node_centor_tmp_list):
            node = onnx.helper.make_node('Add',
                   inputs=[outputs['OutputBox'][0] + "@slice_" + str(1)]\
                       + [name_box_split[count]],
                   outputs=name)
            node_list.append(node)
            count = count + 1
    if not norm:
        inputs_sub = [[name_centor_w_tmp[0], name_box_split[0]],
                      [name_centor_h_tmp[0], name_box_split[1]]]
    else:
        inputs_sub = [[name_box_split[2], name_box_split[0]],
                      [name_box_split[3], name_box_split[1]]]
    outputs_sub = [result_name + "@pb_w", result_name + "@pb_h"]
    for i in range(0, 2):
        node = onnx.helper.make_node(
            'Sub', inputs=inputs_sub[i], outputs=[outputs_sub[i]])
        node_list.append(node)
    # according to prior_box height and weight to get centor x, y 
    name_half_value = [result_name + "@half_value"]
    node_half_value = onnx.helper.make_node(
        'Constant',
        inputs=[],
        outputs=name_half_value,
        value=onnx.helper.make_tensor(
            name=name_slice_indices[i][0] + "@const",
            data_type=onnx.TensorProto.FLOAT,
            dims=(),
            vals=[0.5]))
    node_list.append(node_half_value)
    outputs_half_wh = [[result_name + "@pb_w_half"],
                       [result_name + "@pb_h_half"]]
    inputs_half_wh = [[result_name + "@pb_w", name_half_value[0]],
                      [result_name + "@pb_h", name_half_value[0]]]

    for i in range(0, 2):
        node = onnx.helper.make_node(
            'Mul', inputs=inputs_half_wh[i], outputs=outputs_half_wh[i])
        node_list.append(node)

    inputs_centor_xy = [[outputs_half_wh[0][0], name_slice_x1],
                        [outputs_half_wh[1][0], name_slice_y1]]

    outputs_centor_xy = [[result_name + "@pb_x"], [result_name + "@pb_y"]]

    # final calc the centor x ,y 
    for i in range(0, 2):
        node = onnx.helper.make_node(
            'Add', inputs=inputs_centor_xy[i], outputs=outputs_centor_xy[i])
        node_list.append(node)
    # reshape the data
    shape = (1, split_shape[0]) if axis == 0 else (split_shape[0], 1)

    # need to reshape the data
    inputs_transpose_pb = [
        [result_name + "@pb_w"],
        [result_name + "@pb_h"],
        [result_name + "@pb_x"],
        [result_name + "@pb_y"],
    ]
    outputs_transpose_pb = [
        [result_name + "@pb_w_transpose"],
        [result_name + "@pb_h_transpose"],
        [result_name + "@pb_x_transpose"],
        [result_name + "@pb_y_transpose"],
    ]
    if axis == 0:
        name_reshape_pb = [result_name + "@pb_transpose"]
        # reshape the data 
        for i in range(0, 4):
            node = onnx.helper.make_node(
                'Transpose',
                inputs=inputs_transpose_pb[i],
                outputs=outputs_transpose_pb[i])
            node_list.append(node)
    # decoder the box according to the target_box and variacne  
    name_variance_raw = [result_name + "@variance_raw"]
    name_variance_unsqueeze = [result_name + "@variance_unsqueeze"]
    shape = []
    # make node to extend the data 
    var_split_axis = 0
    var_split_inputs_name = []
    if 'PriorBoxVar' in inputs and len(inputs['PriorBoxVar']) > 0:
        if axis == 1:
            raise Exception(
                "The op box_coder has variable do not support aixs broadcast")
        prior_variance_var = block.vars[get_old_name(inputs['PriorBoxVar'][0])]
        axes = []
        var_split_inputs_name = [result_name + "@variance_split"]
        node = onnx.helper.make_node(
            'Transpose',
            inputs=inputs['PriorBoxVar'],
            outputs=var_split_inputs_name)
        node_list.append(node)
        var_split_axis = 0
    else:
        variances = [1.0, 1.0, 1.0, 1.0]
        if 'variance' in attrs and len(attrs['variance']) > 0:
            variances = [float(var) for var in attrs['variance']]
        node_variance_create = onnx.helper.make_node(
            'Constant',
            inputs=[],
            outputs=name_variance_raw,
            value=onnx.helper.make_tensor(
                name=name_variance_raw[0] + "@const",
                data_type=onnx.TensorProto.FLOAT,
                dims=[len(variances)],
                vals=variances))
        node_list.append(node_variance_create)
        var_split_axis = 0
        var_split_inputs_name = name_variance_raw

    # decode the result 
    outputs_split_variance = [
        result_name + "@variance_split" + str(i) for i in range(0, 4)
    ]
    outputs_split_targebox = [
        result_name + "@targebox_split" + str(i) for i in range(0, 4)
    ]
    node_split_var = onnx.helper.make_node(
        'Split',
        inputs=var_split_inputs_name,
        outputs=outputs_split_variance,
        axis=var_split_axis)
    node_split_target = onnx.helper.make_node(
        'Split',
        inputs=inputs['TargetBox'],
        outputs=outputs_split_targebox,
        axis=2)
    node_list.extend([node_split_var, node_split_target])

    outputs_squeeze_targebox = [
        result_name + "@targebox_squeeze" + str(i) for i in range(0, 4)
    ]
    for (input_name, output_name) in zip(outputs_split_targebox,
                                         outputs_squeeze_targebox):
        node = onnx.helper.make_node(
            'Squeeze', inputs=[input_name], outputs=[output_name], axes=[2])
        node_list.append(node)

    output_shape_step1 = list(t_size)[:-1]

    inputs_tb_step1 = [
        [outputs_squeeze_targebox[0], outputs_split_variance[0]],
        [outputs_squeeze_targebox[1], outputs_split_variance[1]],
        [outputs_squeeze_targebox[2], outputs_split_variance[2]],
        [outputs_squeeze_targebox[3], outputs_split_variance[3]]
    ]
    outputs_tb_step1 = [[result_name + "@decode_x_step1"],
                        [result_name + "@decode_y_step1"],
                        [result_name + "@decode_w_step1"],
                        [result_name + "@decode_h_step1"]]

    for input_step1, output_step_1 in zip(inputs_tb_step1, outputs_tb_step1):
        node = onnx.helper.make_node(
            'Mul', inputs=input_step1, outputs=output_step_1)
        node_list.append(node)
    if axis == 0:
        inputs_tbxy_step2 = [
            [outputs_tb_step1[0][0], outputs_transpose_pb[0][0]],
            [outputs_tb_step1[1][0], outputs_transpose_pb[1][0]]
        ]
    else:
        inputs_tbxy_step2 = [
            [outputs_tb_step1[0][0], inputs_transpose_pb[0][0]],
            [outputs_tb_step1[1][0], inputs_transpose_pb[1][0]]
        ]

    outputs_tbxy_step2 = [[result_name + "@decode_x_step2"],
                          [result_name + "@decode_y_step2"]]

    for input_step2, output_step_2 in zip(inputs_tbxy_step2,
                                          outputs_tbxy_step2):
        node = onnx.helper.make_node(
            'Mul', inputs=input_step2, outputs=output_step_2)
        node_list.append(node)
    if axis == 0:
        inputs_tbxy_step3 = [
            [outputs_tbxy_step2[0][0], outputs_transpose_pb[2][0]],
            [outputs_tbxy_step2[1][0], outputs_transpose_pb[3][0]]
        ]
    else:
        inputs_tbxy_step3 = [
            [outputs_tbxy_step2[0][0], inputs_transpose_pb[2][0]],
            [outputs_tbxy_step2[1][0], inputs_transpose_pb[3][0]]
        ]

    outputs_tbxy_step3 = [[result_name + "@decode_x_step3"],
                          [result_name + "@decode_y_step3"]]

    for input_step3, output_step_3 in zip(inputs_tbxy_step3,
                                          outputs_tbxy_step3):
        node = onnx.helper.make_node(
            'Add', inputs=input_step3, outputs=output_step_3)
        node_list.append(node)

    # deal with width & height
    inputs_tbwh_step2 = [outputs_tb_step1[2], outputs_tb_step1[3]]
    outputs_tbwh_step2 = [[result_name + "@decode_w_step2"],
                          [result_name + "@decode_h_step2"]]

    for input_name, output_name in zip(inputs_tbwh_step2, outputs_tbwh_step2):
        node = onnx.helper.make_node(
            'Exp', inputs=input_name, outputs=output_name)
        node_list.append(node)

    if axis == 0:
        inputs_tbwh_step3 = [
            [outputs_tbwh_step2[0][0], outputs_transpose_pb[0][0]],
            [outputs_tbwh_step2[1][0], outputs_transpose_pb[1][0]]
        ]
    else:
        inputs_tbwh_step3 = [
            [outputs_tbwh_step2[0][0], inputs_transpose_pb[0][0]],
            [outputs_tbwh_step2[1][0], inputs_transpose_pb[1][0]]
        ]

    outputs_tbwh_step3 = [[result_name + "@decode_w_step3"],
                          [result_name + "@decode_h_step3"]]

    for input_name, output_name in zip(inputs_tbwh_step3, outputs_tbwh_step3):
        node = onnx.helper.make_node(
            'Mul', inputs=input_name, outputs=output_name)
        node_list.append(node)

    # final step to calc the result, and concat the result to output 
    # return the output box, [(x1, y1), (x2, y2)]

    inputs_half_tbwh_step4 = [
        [outputs_tbwh_step3[0][0], result_name + "@slice_2"],
        [outputs_tbwh_step3[1][0], result_name + "@slice_2"]
    ]

    outputs_half_tbwh_step4 = [[result_name + "@decode_half_w_step4"],
                               [result_name + "@decode_half_h_step4"]]
    for inputs_name, outputs_name in zip(inputs_half_tbwh_step4,
                                         outputs_half_tbwh_step4):
        node = onnx.helper.make_node(
            'Div', inputs=inputs_name, outputs=outputs_name)
        node_list.append(node)
    inputs_output_point1 = [
        [outputs_tbxy_step3[0][0], outputs_half_tbwh_step4[0][0]],
        [outputs_tbxy_step3[1][0], outputs_half_tbwh_step4[1][0]]
    ]

    outputs_output_point1 = [[result_name + "@ouput_x1"],
                             [result_name + "@output_y1"]]
    for input_name, output_name in zip(inputs_output_point1,
                                       outputs_output_point1):
        node = onnx.helper.make_node(
            'Sub', inputs=input_name, outputs=output_name)
        node_list.append(node)

    inputs_output_point2 = [
        [outputs_tbxy_step3[0][0], outputs_half_tbwh_step4[0][0]],
        [outputs_tbxy_step3[1][0], outputs_half_tbwh_step4[1][0]]
    ]

    outputs_output_point2 = [[result_name + "@ouput_x2"],
                             [result_name + "@output_y2"]]

    for input_name, output_name in zip(inputs_output_point2,
                                       outputs_output_point2):
        node = onnx.helper.make_node(
            'Add', inputs=input_name, outputs=output_name)
        node_list.append(node)
    if not norm:
        inputs_unnorm_point2 = [
            [outputs_output_point2[0][0], result_name + "@slice_1"],
            [outputs_output_point2[1][0], result_name + "@slice_1"]
        ]
        outputs_unnorm_point2 = [[result_name + "@ouput_unnorm_x2"],
                                 [result_name + "@ouput_unnorm_y2"]]

        for input_name, output_name in zip(inputs_unnorm_point2,
                                           outputs_unnorm_point2):
            node = onnx.helper.make_node(
                'Sub', inputs=input_name, outputs=output_name)
            node_list.append(node)
        outputs_output_point2 = outputs_unnorm_point2

    outputs_output_point1.extend(outputs_output_point2)
    ouputs_points_unsqueeze = [[result_name + "@points_unsqueeze_x1"],
                               [result_name + "points_unsqueeze_y1"],
                               [result_name + "points_unsqueeze_x2"],
                               [result_name + "points_unsqueeze_y2"]]

    for input_name, output_name in zip(outputs_output_point1,
                                       ouputs_points_unsqueeze):
        node = onnx.helper.make_node(
            'Unsqueeze',
            inputs=input_name,
            outputs=output_name,
            axes=[len(output_shape_step1)])
        node_list.append(node)
    outputs_points_unsqueeze_list = [
        output[0] for output in ouputs_points_unsqueeze
    ]
    node_point_final = onnx.helper.make_node(
        'Concat',
        inputs=outputs_points_unsqueeze_list,
        outputs=outputs['OutputBox'],
        axis=len(output_shape_step1))
    node_list.append(node_point_final)
    return tuple(node_list)
