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

import onnx
import numpy as np
from onnx import onnx_pb, helper

MAX_FLOAT32 = np.asarray(
    [255, 255, 127, 127], dtype=np.uint8).view(np.float32)[0]


def get_old_name(arg, name_prefix=''):
    prefix_index = arg.find(name_prefix)

    if prefix_index != -1:
        last_prefix = arg[len(name_prefix):]
    else:
        last_prefix = arg
    idx = last_prefix.find('@')
    if idx != -1:
        last_prefix = last_prefix[:idx]
    return name_prefix + last_prefix


def is_static_shape(shape):
    if len(shape) > 1 and shape.count(-1) > 1:
        raise Exception(
            "Converting this model to ONNX need with static input shape, please fix input shape of this model, see doc Q2 in https://github.com/PaddlePaddle/X2Paddle/blob/develop/FAQ.md."
        )


def yolo_box(op, block):
    inputs = dict()
    outputs = dict()
    attrs = dict()
    for name in op.input_names:
        inputs[name] = op.input(name)
    for name in op.output_names:
        outputs[name] = op.output(name)
    for name in op.attr_names:
        attrs[name] = op.attr(name)
    model_name = outputs['Boxes'][0]
    input_shape = block.vars[get_old_name(inputs['X'][0])].shape
    is_static_shape(input_shape)
    image_size = inputs['ImgSize']
    input_height = input_shape[2]
    input_width = input_shape[3]

    class_num = attrs['class_num']
    anchors = attrs['anchors']
    num_anchors = int(len(anchors)) // 2
    downsample_ratio = attrs['downsample_ratio']
    input_size = input_height * downsample_ratio
    conf_thresh = attrs['conf_thresh']
    conf_thresh_mat = np.ones([num_anchors * input_height *
                               input_width]) * conf_thresh

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

    name_number_two = [model_name + "@number_two"]
    node_number_two = onnx.helper.make_node(
        "Constant",
        inputs=[],
        outputs=name_number_two,
        value=onnx.helper.make_tensor(
            name=name_number_two[0] + "@const",
            data_type=onnx.TensorProto.FLOAT,
            dims=(),
            vals=[2]))
    node_list.append(node_number_two)

    outputs_half_w = [model_name + "@half_w"]
    node_half_w = onnx.helper.make_node(
        "Div",
        inputs=outputs_pred_box_w + name_number_two,
        outputs=outputs_half_w)
    node_list.append(node_half_w)

    outputs_half_h = [model_name + "@half_h"]
    node_half_h = onnx.helper.make_node(
        "Div",
        inputs=outputs_pred_box_h + name_number_two,
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
        max=float(MAX_FLOAT32))
    node_list.append(node_pred_box_x1_clip)

    node_pred_box_y1_clip = onnx.helper.make_node(
        'Clip',
        inputs=outputs_pred_box_y1_decode,
        outputs=outputs_pred_box_y1_clip,
        min=0.0,
        max=float(MAX_FLOAT32))
    node_list.append(node_pred_box_y1_clip)

    node_pred_box_x2_clip = onnx.helper.make_node(
        'Clip',
        inputs=outputs_pred_box_x2_sub_w,
        outputs=outputs_pred_box_x2_clip,
        min=0.0,
        max=float(MAX_FLOAT32))
    node_list.append(node_pred_box_x2_clip)

    node_pred_box_y2_clip = onnx.helper.make_node(
        'Clip',
        inputs=outputs_pred_box_y2_sub_h,
        outputs=outputs_pred_box_y2_clip,
        min=0.0,
        max=float(MAX_FLOAT32))
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
    return node_list
