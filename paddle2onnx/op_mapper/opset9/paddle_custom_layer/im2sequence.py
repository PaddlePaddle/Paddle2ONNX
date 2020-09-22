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

im2seq_counter = 0


def im2sequence(op, block):
    global im2sequence_counter
    n, c, h, w = block.var(op.input('X')[0]).shape
    assert h > 0 and w > 0, "Only supported fixed input shape for im2sequence operator."
    stride_h, stride_w = op.attr('strides')
    paddings = op.attr('paddings')
    assert op.attr(
        'out_stride'
    ) != 1, "Only out_stride==1 is supported for im2sequence operator."
    h = h + paddings[0] + paddings[1]
    w = w + paddings[1] + paddings[2]
    kernel_h, kernel_w = op.attr('kernels')
    out_h = 1 + (h - kernel_h + stride_h - 1) // stride_h
    out_w = 1 + (w - kernel_w + stride_w - 1) // stride_w
    h_steps = list()
    for i in range(out_h):
        h_steps.append([i * stride_h, i * stride_h + kernel_h])
    w_steps = list()
    for i in range(out_w):
        w_steps.append([i * stride_w, i * stride_w + kernel_w])

    nodes = list()
    slice_blocks = list()
    for i in range(out_h):
        for j in range(out_w):
            starts_name = "im2sequence.starts.{}.{}.{}".format(im2seq_counter,
                                                               i, j)
            starts_tensor = helper.make_tensor(
                name=starts_name,
                data_type=onnx_pb.TensorProto.INT64,
                dims=[4],
                vals=[0, 0, h_steps[i][0], w_steps[j][0]])
            ends_name = "im2sequence.ends.{}.{}.{}".format(im2seq_counter, i, j)
            ends_tensor = helper.make_tensor(
                name=ends_name,
                data_type=onnx_pb.TensorProto.INT64,
                dims=[4],
                vals=[999999, 999999, h_steps[i][1], w_steps[j][1]])
            starts_node = helper.make_node(
                'Constant',
                inputs=[],
                outputs=[starts_name],
                value=starts_tensor)
            ends_node = helper.make_node(
                'Constant', inputs=[], outputs=[ends_name], value=ends_tensor)
            nodes.extend([starts_node, ends_node])

            slice_block_name = "im2sequence.slice.{}.{}.{}".format(
                im2seq_counter, i, j)
            slice_block_node = helper.make_node(
                'Slice',
                inputs=[op.input('X')[0], starts_name, ends_name],
                outputs=[slice_block_name])
            flatten_block_name = "im2sequence.flatten.{}.{}.{}".format(
                im2seq_counter, i, j)
            flatten_block_node = helper.make_node(
                "Flatten",
                inputs=[slice_block_name],
                outputs=[flatten_block_name],
                axis=0)
            nodes.extend([slice_block_node, flatten_block_node])
            slice_blocks.append(flatten_block_name)
    concat_block_name = "im2sequence.concat_block.{}".format(im2seq_counter)
    #    concat_block_node = helper.make_node("Concat", inputs=slice_blocks, outputs=[concat_block_name], axis=0)
    concat_block_node = helper.make_node(
        "Concat", inputs=slice_blocks, outputs=op.output('Out'), axis=0)
    nodes.append(concat_block_node)
    print("\n\n==========Importance Notice===========")
    print(
        "Since im2sequence operator is used in your paddlepaddle model, the translated onnx model only support input data with batch_size=1."
    )
    print("======================================\n")
    return nodes
