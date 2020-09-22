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

import math
import sys
import paddle2onnx
import os
import numpy as np
import paddle.fluid.core as core
import paddle.fluid as fluid
import onnx
from onnx import helper, onnx_pb
from paddle2onnx.op_mapper.opset10.opset import OpSet10


class OpSet11(OpSet10):
    def __init__(self):
        super(OpSet11, self).__init__()

    def relu6(self, op, block):
        min_name = self.get_name(op.type, 'min')
        max_name = self.get_name(op.type, 'max')
        min_node = self.make_constant_node(min_name, onnx_pb.TensorProto.FLOAT,
                                           0)
        max_node = self.make_constant_node(max_name, onnx_pb.TensorProto.FLOAT,
                                           op.attr('threshold'))
        node = helper.make_node(
            'Clip',
            inputs=[op.input('X')[0], min_name, max_name],
            outputs=op.output('Out'), )
        return [min_node, max_node, node]

    def pad2d(self, op, block):
        x_shape = block.var(op.input('X')[0]).shape
        paddings = op.attr('paddings')
        onnx_pads = []
        #TODO support pads is Variable
        if op.attr('data_format') == 'NCHW':
            pads = [
                0, 0, paddings[0], paddings[2], 0, 0, paddings[1], paddings[3]
            ]
        else:
            pads = [
                0, paddings[0], paddings[2], 0, 0, paddings[1], paddings[3], 0
            ]
        pads_name = self.get_name(op.type, 'pads')
        pads_node = self.make_constant_node(pads_name,
                                            onnx_pb.TensorProto.INT64, pads)
        constant_value_name = self.get_name(op.type, 'constant_value')
        constant_value_node = self.make_constant_node(constant_value_name,
                                                      onnx_pb.TensorProto.FLOAT,
                                                      op.attr('pad_value'))
        node = helper.make_node(
            'Pad',
            inputs=op.input('X') + [pads_name, constant_value_name],
            outputs=op.output('Out'),
            mode=op.attr('mode'))
        return [pads_node, constant_value_node, node]

    def clip(self, op, block):
        min_name = self.get_name(op.type, 'min')
        max_name = self.get_name(op.type, 'max')
        min_node = self.make_constant_node(min_name, onnx_pb.TensorProto.FLOAT,
                                           op.attr('min'))
        max_node = self.make_constant_node(max_name, onnx_pb.TensorProto.FLOAT,
                                           op.attr('max'))
        node = helper.make_node(
            'Clip',
            inputs=[op.input('X')[0], min_name, max_name],
            outputs=op.output('Out'))
        return [min_node, max_node, node]

    def bilinear_interp(self, op, block):
        input_names = op.input_names
        coordinate_transformation_mode = ''
        align_corners = op.attr('align_corners')
        align_mode = op.attr('align_mode')
        if align_corners:
            coordinate_transformation_mode = 'align_corners'
        elif align_mode == 1:
            coordinate_transformation_mode = 'asymmetric'
        else:
            coordinate_transformation_mode = 'half_pixel'

        roi_name = self.get_name(op.type, 'roi')
        roi_node = self.make_constant_node(roi_name, onnx_pb.TensorProto.FLOAT,
                                           [1, 1, 1, 1, 1, 1, 1, 1])
        if ('OutSize' in input_names and len(op.input('OutSize')) > 0) or (
                'SizeTensor' in input_names and
                len(op.input('SizeTensor')) > 0):
            node_list = list()
            empty_name = self.get_name(op.type, 'empty')
            empty_tensor = helper.make_tensor(
                empty_name,
                onnx_pb.TensorProto.FLOAT, (0, ),
                np.array([]).astype('float32'),
                raw=False)
            empty_node = helper.make_node(
                'Constant', [], outputs=[empty_name], value=empty_tensor)
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
            node_list.extend([
                roi_node, empty_node, shape_node0, starts_node, ends_node,
                shape_node1
            ])
            if 'OutSize' in input_names and len(op.input('OutSize')) > 0:
                cast_shape_name = self.get_name(op.type, "shape.cast")
                cast_shape_node = helper.make_node(
                    'Cast',
                    inputs=op.input('OutSize'),
                    outputs=[cast_shape_name],
                    to=onnx_pb.TensorProto.INT64)
                node_list.append(cast_shape_node)
            else:
                concat_shape_name = self.get_name(op.type, "shape.concat")
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
            shape_name3 = self.get_name(op.type, "shape.concat")
            shape_node3 = helper.make_node(
                'Concat',
                inputs=[shape_name1, cast_shape_name],
                outputs=[shape_name3],
                axis=0)
            result_node = helper.make_node(
                'Resize',
                inputs=[op.input('X')[0], roi_name, empty_name, shape_name3],
                outputs=op.output('Out'),
                mode='linear',
                coordinate_transformation_mode=coordinate_transformation_mode)
            node_list.extend([shape_node3, result_node])
            return node_list
        elif 'Scale' in input_names and len(op.input('Scale')) > 0:
            node = helper.make_node(
                'Resize',
                inputs=[op.input('X')[0], roi_name, op.input('Scale')[0]],
                outputs=op.output('Out'),
                mode='linear',
                coordinate_transformation_mode=coordinate_transformation_mode)
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
                    inputs=[op.input('X')[0], roi_name, scale_name],
                    outputs=op.output('Out'),
                    mode='nearest',
                    coordinate_transformation_mode=coordinate_transformation_mode
                )
                return [scale_node, roi_node, node]
            else:
                raise Exception("Unexpected situation happend")
        return [roi_node, node]

    def nearest_interp(self, op, block):
        input_names = op.input_names
        coordinate_transformation_mode = ''
        align_corners = op.attr('align_corners')
        if align_corners:
            coordinate_transformation_mode = 'align_corners'
        else:
            coordinate_transformation_mode = 'half_pixel'
        roi_name = self.get_name(op.type, 'roi')
        roi_node = self.make_constant_node(roi_name, onnx_pb.TensorProto.FLOAT,
                                           [1, 1, 1, 1, 1, 1, 1, 1])
        if 'OutSize' in input_names and len(op.input('OutSize')) > 0:
            node = helper.make_node(
                'Resize',
                inputs=[op.input('X')[0], roi_name, op.input('OutSize')[0]],
                outputs=op.output('Out'),
                mode='nearest',
                coordinate_transformation_mode=coordinate_transformation_mode)
        elif 'Scale' in input_names and len(op.input('Scale')) > 0:
            node = helper.make_node(
                'Resize',
                inputs=[op.input('X')[0], roi_name, op.input('Scale')[0]],
                outputs=op.output('Out'),
                mode='nearest',
                coordinate_transformation_mode=coordinate_transformation_mode)
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
                    inputs=[op.input('X')[0], roi_name, scale_name],
                    outputs=op.output('Out'),
                    mode='nearest',
                    coordinate_transformation_mode=coordinate_transformation_mode
                )
                return [scale_node, roi_node, node]
            else:
                raise Exception("Unexpected situation happend")
        return [roi_node, node]

    def hard_swish(self, op, block):
        min_name = self.get_name(op.type, 'min')
        max_name = self.get_name(op.type, 'max')
        scale_name = self.get_name(op.type, 'scale')
        offset_name = self.get_name(op.type, 'offset')
        min_node = self.make_constant_node(min_name, onnx_pb.TensorProto.FLOAT,
                                           0)
        max_node = self.make_constant_node(max_name, onnx_pb.TensorProto.FLOAT,
                                           op.attr('threshold'))
        scale_node = self.make_constant_node(scale_name,
                                             onnx_pb.TensorProto.FLOAT,
                                             op.attr('scale'))
        offset_node = self.make_constant_node(offset_name,
                                              onnx_pb.TensorProto.FLOAT,
                                              op.attr('offset'))

        name0 = self.get_name(op.type, 'add')
        node0 = helper.make_node(
            'Add', inputs=[op.input('X')[0], offset_name], outputs=[name0])
        name1 = self.get_name(op.type, 'relu')
        node1 = helper.make_node(
            'Clip',
            inputs=[name0, min_name, max_name],
            outputs=[name1], )
        name2 = self.get_name(op.type, 'mul')
        node2 = helper.make_node(
            'Mul', inputs=[op.input('X')[0], name1], outputs=[name2])
        node3 = helper.make_node(
            'Div', inputs=[name2, scale_name], outputs=op.output('Out'))
        return [
            min_node, max_node, scale_node, offset_node, node0, node1, node2,
            node3
        ]

    def yolo_box(self, op, block):
        from .paddle_custom_layer.yolo_box import yolo_box
        return yolo_box(op, block)

    def multiclass_nms(self, op, block):
        from .paddle_custom_layer.multiclass_nms import multiclass_nms
        return multiclass_nms(op, block)
