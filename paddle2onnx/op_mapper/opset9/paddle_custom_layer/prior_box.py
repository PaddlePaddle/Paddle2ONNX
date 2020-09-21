# Copyright (c) 2019  PaddlePaddle Authors. All Rights Reserved.
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
from onnx import helper
import paddle.fluid as fluid
import paddle.fluid.core as core


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


def prior_box(op, block):
    """
    In this function, use the attribute to get the prior box, because we do not use
    the image data and feature map, wo could the python code to create the varaible,
    and to create the onnx tensor as output.
    """
    flip = bool(op.attr('flip'))
    clip = bool(op.attr('clip'))
    min_max_aspect_ratios_order = bool(op.attr('min_max_aspect_ratios_order'))
    min_sizes = [float(size) for size in op.attr('min_sizes')]
    max_sizes = [float(size) for size in op.attr('max_sizes')]
    if isinstance(op.attr('aspect_ratios'), list):
        aspect_ratios = [float(ratio) for ratio in op.attr('aspect_ratios')]
    else:
        aspect_ratios = [float(op.attr('aspect_ratios'))]
    variances = [float(var) for var in op.attr('variances')]
    # set min_max_aspect_ratios_order = false
    output_ratios = ExpandAspectRations(aspect_ratios, flip)

    step_w = float(op.attr('step_w'))
    step_h = float(op.attr('step_h'))
    offset = float(op.attr('offset'))

    input_shape = block.var(op.input('Input')[0]).shape
    image_shape = block.var(op.input('Image')[0]).shape

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
        outputs=op.output('Boxes'),
        value=onnx.helper.make_tensor(
            name=op.output('Boxes')[0] + "@const",
            data_type=onnx.TensorProto.FLOAT,
            dims=out_boxes.shape,
            vals=out_boxes.flatten()))
    node_vars = onnx.helper.make_node(
        'Constant',
        inputs=[],
        outputs=op.output('Variances'),
        value=onnx.helper.make_tensor(
            name=op.output('Variances')[0] + "@const",
            data_type=onnx.TensorProto.FLOAT,
            dims=out_var.shape,
            vals=out_var.flatten()))
    return [node_boxes, node_vars]
