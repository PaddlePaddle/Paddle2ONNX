#   Copyright (c) 2019  PaddlePaddle Authors. All Rights Reserved.
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

import math
import sys
import paddle2onnx
import os
import numpy as np
import paddle.fluid.core as core
import paddle.fluid as fluid
import onnx
from onnx import helper, onnx_pb
from paddle2onnx.op_mapper.opset9.opset import OpSet9


class OpSet10(OpSet9):
    def __init__(self):
        super(OpSet10, self).__init__()

    def slice(self, op, block):
        axes = op.attr('axes')
        starts = op.attr('starts')
        ends = op.attr('ends')
        axes_name = self.get_name(op.type, 'axes')
        starts_name = self.get_name(op.type, 'starts')
        ends_name = self.get_name(op.type, 'ends')

        axes_node = self.make_constant_node(axes_name,
                                            onnx_pb.TensorProto.INT64, axes)
        starts_node = self.make_constant_node(starts_name,
                                              onnx_pb.TensorProto.INT64, starts)
        ends_node = self.make_constant_node(ends_name,
                                            onnx_pb.TensorProto.INT64, ends)
        node = helper.make_node(
            "Slice",
            inputs=[op.input('Input')[0], starts_name, ends_name, axes_name],
            outputs=op.output('Out'), )
        return [starts_node, ends_node, axes_node, node]

    def bilinear_interp(self, op, block):
        input_names = op.input_names
        input_shape = block.vars[op.input('X')[0]].shape
        if op.attr('align_corners') or op.attr('align_mode') == 0:
            raise Exception(
                "Resize in onnx(opset<=10) only support coordinate_transformation_mode: 'asymmetric', Try converting with --onnx_opset 11"
            )
        if ('OutSize' in input_names and len(op.input('OutSize')) > 0) or (
                'SizeTensor' in input_names and
                len(op.input('SizeTensor')) > 0):
            node_list = list()
            shape_name0 = self.get_name(op.type, 'shape')
            shape_node0 = helper.make_node(
                'Shape', inputs=op.input('X'), outputs=[shape_name0])
            starts_name = self.get_name(op.type, 'slice.starts')
            starts_node = self.make_constant_node(
                starts_name, onnx_pb.TensorProto.INT64, [0])
            ends_name = self.get_name(op.type, 'slice.ends')
            ends_node = self.make_constant_node(ends_name,
                                                onnx_pb.TensorProto.INT64, [2])
            shape_name1 = self.get_name(op.type, 'shape')
            shape_node1 = helper.make_node(
                'Slice',
                inputs=[shape_name0, starts_name, ends_name],
                outputs=[shape_name1])
            node_list.extend([shape_node0, starts_node, ends_node, shape_node1])
            if 'OutSize' in input_names and len(op.input('OutSize')) > 0:
                cast_shape_name = self.get_name(op.type, "shape.cast")
                cast_shape_node = helper.make_node(
                    'Cast',
                    inputs=op.input('OutSize'),
                    outputs=[cast_shape_name],
                    to=onnx_pb.TensorProto.INT64)
                node_list.append(cast_shape_node)
            else:
                concat_shape_name = self.get_name(
                    op.type, op.output('Out')[0] + "shape.concat")
                concat_shape_node = helper.make_node(
                    "Concat",
                    inputs=op.input('SizeTensor'),
                    outputs=[concat_shape_name],
                    axis=0)
                cast_shape_name = self.get_name(op.type, "shape.cast")
                cast_shape_node = helper.make_node(
                    'Cast',
                    inputs=[concat_shape_name],
                    outputs=[cast_shape_name],
                    to=onnx_pb.TensorProto.INT64)
                node_list.extend([concat_shape_node, cast_shape_node])
            shape_name2 = self.get_name(op.type, "shape.concat")
            shape_node2 = helper.make_node(
                'Concat',
                inputs=[shape_name1, cast_shape_name],
                outputs=[shape_name2],
                axis=0)
            node_list.append(shape_node2)
            cast_shape_name2 = self.get_name(op.type, "shape.cast")
            cast_shape_node2 = helper.make_node(
                'Cast',
                inputs=[shape_name2],
                outputs=[cast_shape_name2],
                to=onnx_pb.TensorProto.FLOAT)
            node_list.append(cast_shape_node2)
            cast_shape_name0 = self.get_name(op.type, "shape.cast")
            cast_shape_node0 = helper.make_node(
                'Cast',
                inputs=[shape_name0],
                outputs=[cast_shape_name0],
                to=onnx_pb.TensorProto.FLOAT)
            node_list.append(cast_shape_node0)
            outputs_h_w_scales = op.output('Out')[0] + "@out_hw_scales"
            node_h_w_scales = helper.make_node(
                'Div',
                inputs=[cast_shape_name2, cast_shape_name0],
                outputs=[outputs_h_w_scales])
            node_list.append(node_h_w_scales)
            result_node = helper.make_node(
                'Resize',
                inputs=[op.input('X')[0], outputs_h_w_scales],
                outputs=op.output('Out'),
                mode='linear')
            node_list.extend([result_node])
            return node_list
        elif 'Scale' in input_names and len(op.input('Scale')) > 0:
            node = helper.make_node(
                'Resize',
                inputs=[op.input('X')[0], op.input('Scale')[0]],
                outputs=op.output('Out'),
                mode='linear')
        else:
            out_shape = [op.attr('out_h'), op.attr('out_w')]
            scale = op.attr('scale')
            if out_shape.count(-1) > 0:
                scale_name = self.get_name(op.type, 'scale')
                scale_node = self.make_constant_node(scale_name,
                                                     onnx_pb.TensorProto.FLOAT,
                                                     [1, 1, scale, scale])
                node = helper.make_node(
                    'Resize',
                    inputs=[op.input('X')[0], scale_name],
                    outputs=op.output('Out'),
                    mode='linear')
                return [scale_node, node]
            else:
                raise Exception("Unexpected situation happend")
        return node

    def nearest_interp(self, op, block):
        input_names = op.input_names
        if op.attr('align_corners'):
            raise Exception(
                "Resize in onnx(opset<=10) only support coordinate_transformation_mode: 'asymmetric', Try converting with --onnx_opset 11"
            )
        if 'OutSize' in input_names and len(op.input('OutSize')) > 0:
            node_list = list()
            shape_name0 = self.get_name(op.type, 'shape')
            shape_node0 = helper.make_node(
                'Shape', inputs=op.input('X'), outputs=[shape_name0])
            starts_name = self.get_name(op.type, 'slice.starts')
            starts_node = self.make_constant_node(
                starts_name, onnx_pb.TensorProto.INT64, [0])
            ends_name = self.get_name(op.type, 'slice.ends')
            ends_node = self.make_constant_node(ends_name,
                                                onnx_pb.TensorProto.INT64, [2])
            shape_name1 = self.get_name(op.type, 'shape')
            shape_node1 = helper.make_node(
                'Slice',
                inputs=[shape_name0, starts_name, ends_name],
                outputs=[shape_name1])
            node_list.extend([shape_node0, starts_node, ends_node, shape_node1])
            if 'OutSize' in input_names and len(op.input('OutSize')) > 0:
                cast_shape_name = self.get_name(op.type, "shape.cast")
                cast_shape_node = helper.make_node(
                    'Cast',
                    inputs=op.input('OutSize'),
                    outputs=[cast_shape_name],
                    to=onnx_pb.TensorProto.INT64)
                node_list.append(cast_shape_node)
            else:
                concat_shape_name = self.get_name(
                    op.type, op.output('Out')[0] + "shape.concat")
                concat_shape_node = helper.make_node(
                    "Concat",
                    inputs=op.input('SizeTensor'),
                    outputs=[concat_shape_name],
                    axis=0)
                cast_shape_name = self.get_name(op.type, "shape.cast")
                cast_shape_node = helper.make_node(
                    'Cast',
                    inputs=[concat_shape_name],
                    outputs=[cast_shape_name],
                    to=onnx_pb.TensorProto.INT64)
                node_list.extend([concat_shape_node, cast_shape_node])
            shape_name2 = self.get_name(op.type, "shape.concat")
            shape_node2 = helper.make_node(
                'Concat',
                inputs=[shape_name1, cast_shape_name],
                outputs=[shape_name2],
                axis=0)
            node_list.append(shape_node2)
            cast_shape_name2 = self.get_name(op.type, "shape.cast")
            cast_shape_node2 = helper.make_node(
                'Cast',
                inputs=[shape_name2],
                outputs=[cast_shape_name2],
                to=onnx_pb.TensorProto.FLOAT)
            node_list.append(cast_shape_node2)
            cast_shape_name0 = self.get_name(op.type, "shape.cast")
            cast_shape_node0 = helper.make_node(
                'Cast',
                inputs=[shape_name0],
                outputs=[cast_shape_name0],
                to=onnx_pb.TensorProto.FLOAT)
            node_list.append(cast_shape_node0)
            outputs_h_w_scales = op.output('Out')[0] + "@out_hw_scales"
            node_h_w_scales = helper.make_node(
                'Div',
                inputs=[cast_shape_name2, cast_shape_name0],
                outputs=[outputs_h_w_scales])
            node_list.append(node_h_w_scales)
            result_node = helper.make_node(
                'Resize',
                inputs=[op.input('X')[0], outputs_h_w_scales],
                outputs=op.output('Out'),
                mode='linear')
            node_list.extend([result_node])
            return node_list
        elif 'Scale' in input_names and len(op.input('Scale')) > 0:
            node = helper.make_node(
                'Resize',
                inputs=[op.input('X')[0], op.input('Scale')[0]],
                outputs=op.output('Out'),
                mode='nearest')
        else:
            out_shape = [op.attr('out_h'), op.attr('out_w')]
            scale = op.attr('scale')
            if out_shape.count(-1) > 0:
                scale_name = self.get_name(op.type, 'scale')
                scale_node = self.make_constant_node(scale_name,
                                                     onnx_pb.TensorProto.FLOAT,
                                                     [1, 1, scale, scale])
                node = helper.make_node(
                    'Resize',
                    inputs=[op.input('X')[0], scale_name],
                    outputs=op.output('Out'),
                    mode='nearest')
                return [scale_node, node]
            else:
                raise Exception("Unexpected situation happend")
        return node
