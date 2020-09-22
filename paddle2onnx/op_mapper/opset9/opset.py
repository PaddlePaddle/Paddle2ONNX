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
import os
import numpy as np
import paddle.fluid.core as core
import paddle.fluid as fluid
import onnx
from onnx import helper, onnx_pb


class OpSet9(object):
    def __init__(self):
        self.paddle_onnx_dtype_map = {
            core.VarDesc.VarType.FP32: onnx_pb.TensorProto.FLOAT,
            core.VarDesc.VarType.FP64: onnx_pb.TensorProto.DOUBLE,
            core.VarDesc.VarType.INT32: onnx_pb.TensorProto.INT32,
            core.VarDesc.VarType.INT16: onnx_pb.TensorProto.INT16,
            core.VarDesc.VarType.INT16: onnx_pb.TensorProto.UINT16,
            core.VarDesc.VarType.INT64: onnx_pb.TensorProto.INT64,
            core.VarDesc.VarType.BOOL: onnx_pb.TensorProto.BOOL
        }
        self.name_counter = dict()

    def get_name(self, op_name, var_name):
        name = 'p2o.{}.{}'.format(op_name, var_name)
        if name not in self.name_counter:
            self.name_counter[name] = 0
        else:
            self.name_counter[name] += 1
        return name + '.{}'.format(self.name_counter[name])

    def make_constant_node(self, name, dtype, value=None):
        if isinstance(value, list):
            dims = (len(value), )
        elif value is None:
            dims = ()
            value = []
        else:
            dims = ()
            value = [value]
        tensor = helper.make_tensor(
            name=name, data_type=dtype, dims=dims, vals=value)
        node = helper.make_node(
            'Constant', inputs=[], outputs=[name], value=tensor)
        return node

    def convert_weights(self, program, scope=None):
        var_names = program.global_block().vars
        nodes = list()
        for name in var_names:
            var = program.global_block().var(name)
            if name.endswith('feed') or name.endswith('fetch'):
                continue
            if not var.persistable:
                continue
            weight = np.array(scope.find_var(name).get_tensor())
            tensor = helper.make_tensor(
                name=name,
                dims=var.shape,
                data_type=self.paddle_onnx_dtype_map[var.dtype],
                vals=weight.flatten().tolist())
            node = helper.make_node(
                'Constant', inputs=[], outputs=[name], value=tensor)
            nodes.append(node)
        return nodes

    def conv2d(self, op, block):
        kernel_shape = block.var(op.input('Filter')[0]).shape
        node = helper.make_node(
            'Conv',
            inputs=op.input('Input') + op.input('Filter'),
            outputs=op.output('Output'),
            dilations=op.attr('dilations'),
            kernel_shape=kernel_shape[-2:],
            strides=op.attr('strides'),
            group=op.attr('groups'),
            pads=op.attr('paddings') + op.attr('paddings'))
        return node

    def conv2d_transpose(self, op, block):
        kernel_shape = block.var(op.input('Filter')[0]).shape
        node = helper.make_node(
            'ConvTranspose',
            inputs=op.input('Input') + op.input('Filter'),
            outputs=op.output('Output'),
            dilations=op.attr('dilations'),
            kernel_shape=kernel_shape[-2:],
            strides=op.attr('strides'),
            group=1,
            pads=op.attr('paddings') + op.attr('paddings'))
        return node

    def relu(self, op, block):
        node = helper.make_node(
            'Relu', inputs=op.input('X'), outputs=op.output('Out'))
        return node

    def prelu(self, op, block):
        node = helper.make_node(
            'PRelu',
            inputs=[op.input('X')[0], op.input('Alpha')[0]],
            outputs=op.output('Out'))
        return node

    def tanh(self, op, block):
        node = helper.make_node(
            'Tanh', inputs=op.input('X'), outputs=op.output('Out'))
        return node

    def log(self, op, block):
        node = helper.make_node(
            'Log', inputs=op.input('X'), outputs=op.output('Out'))
        return node

    def sigmoid(self, op, block):
        node = helper.make_node(
            'Sigmoid', inputs=op.input('X'), outputs=op.output('Out'))
        return node

    def clip(self, op, block):
        min_value = op.attr('min')
        max_value = op.attr('max')
        node = helper.make_node(
            'Clip',
            inputs=[op.input('X')[0]],
            outputs=op.output('Out'),
            max=max_value,
            min=min_value)
        return node

    def exp(self, op, block):
        node = helper.make_node(
            'Exp', inputs=op.input('X'), outputs=op.output('Out'))
        return node

    def abs(self, op, block):
        node = helper.make_node(
            'Abs', inputs=op.input('X'), outputs=op.output('Out'))
        return node

    def leaky_relu(self, op, block):
        node = helper.make_node(
            'LeakyRelu',
            inputs=op.input('X'),
            outputs=op.output('Out'),
            alpha=op.attr('alpha'))
        return node

    def elementwise_add(self, op, block):
        axis = op.attr('axis')
        x_shape = block.var(op.input('X')[0]).shape
        y_shape = block.var(op.input('Y')[0]).shape
        if len(y_shape) == 1 and axis == 1:
            shape_name = self.get_name(op.type, 'shape')
            shape_value = [1] * len(x_shape)
            shape_value[axis] = y_shape[0]
            shape_node = self.make_constant_node(
                shape_name, onnx_pb.TensorProto.INT64, shape_value)
            temp_value = self.get_name(op.type, 'temp')
            y_node = helper.make_node(
                'Reshape',
                inputs=[op.input('Y')[0], shape_name],
                outputs=[temp_value])
            node = helper.make_node(
                'Add',
                inputs=[op.input('X')[0], temp_value],
                outputs=op.output('Out'))
            return [shape_node, y_node, node]
        elif axis == -1 or axis == (len(x_shape) - 1
                                    ) or len(x_shape) == len(y_shape):
            node = helper.make_node(
                'Add',
                inputs=[op.input('X')[0], op.input('Y')[0]],
                outputs=op.output('Out'))
            return node
        else:
            raise Exception("Unexpected situation happend in elementwise_add")

    def elementwise_sub(self, op, block):
        axis = op.attr('axis')
        x_shape = block.var(op.input('X')[0]).shape
        y_shape = block.var(op.input('Y')[0]).shape
        if len(y_shape) == 1 and axis == 1:
            shape_name = self.get_name(op.type, 'shape')
            shape_value = [1] * len(x_shape)
            shape_value[axis] = y_shape[0]
            shape_node = self.make_constant_node(
                shape_name, onnx_pb.TensorProto.INT64, shape_value)
            temp_value = self.get_name(op.type, 'temp')
            y_node = helper.make_node(
                'Reshape',
                inputs=[op.input('Y')[0], shape_name],
                outputs=[temp_value])
            node = helper.make_node(
                'Sub',
                inputs=[op.input('X')[0], temp_value],
                outputs=op.output('Out'))
            return [shape_node, y_node, node]
        elif axis == -1 or axis == (len(x_shape) - 1
                                    ) or len(x_shape) == len(y_shape):
            node = helper.make_node(
                'Sub',
                inputs=[op.input('X')[0], op.input('Y')[0]],
                outputs=op.output('Out'))
            return node
        else:
            raise Exception("Unexpected situation happend in elementwise_sub")

    def pool2d(self, op, block):
        pool_type = {
            'max': ('MaxPool', 'GlobalMaxPool'),
            'avg': ('AveragePool', 'GlobalAveragePool')
        }
        if op.attr('global_pooling'):
            node = helper.make_node(
                pool_type[op.attr('pooling_type')][1],
                inputs=op.input('X'),
                outputs=op.output('Out'), )
        elif op.attr('adaptive'):
            raise Excpetion("ONNX cannot support adaptive pool")
        else:
            input_shape = block.var(op.input('X')[0]).shape
            k_size = op.attr('ksize')
            paddings = op.attr('paddings')
            if input_shape[2] > 0 and input_shape[2] + paddings[0] < k_size[0]:
                k_size[0] = input_shape[2] + paddings[0]
            if input_shape[3] > 0 and input_shape[3] + paddings[1] < k_size[1]:
                k_size[1] = input_shape[3] + paddings[1]
            node = helper.make_node(
                pool_type[op.attr('pooling_type')][0],
                inputs=op.input('X'),
                outputs=op.output('Out'),
                kernel_shape=k_size,
                strides=op.attr('strides'),
                pads=op.attr('paddings') + op.attr('paddings'))
        return node

    def pad2d(self, op, block):
        x_shape = block.var(op.input('X')[0]).shape
        paddings = op.attr('paddings')
        onnx_pads = []
        if op.attr('data_format') == 'NCHW':
            pads = [
                0, 0, paddings[0], paddings[2], 0, 0, paddings[1], paddings[3]
            ]
        else:
            pads = [
                0, paddings[0], paddings[2], 0, 0, paddings[1], paddings[3], 0
            ]
        #TODO support pads is Variable
        node = helper.make_node(
            'Pad',
            inputs=op.input('X'),
            outputs=op.output('Out'),
            mode=op.attr('mode'),
            value=op.attr('pad_value'),
            pads=pads)
        return node

    def softmax(self, op, block):
        axis = op.attr('axis')
        shape = block.var(op.output('Out')[0]).shape
        if axis < 0:
            axis += len(shape)
        if axis == len(shape) - 1:
            node = helper.make_node(
                'Softmax',
                inputs=op.input('X'),
                outputs=op.output('Out'),
                axis=op.attr('axis'))
            return node
        else:
            perm = [i for i in range(len(shape))]
            perm[-1] = axis
            perm[axis] = len(shape) - 1
            transpose_name0 = self.get_name(op.type, 'transpose')
            transpose_node0 = helper.make_node(
                'Transpose',
                inputs=op.input('X'),
                outputs=[transpose_name0],
                perm=perm)
            softmax_name = self.get_name(op.type, 'softmax')
            softmax_node = helper.make_node(
                'Softmax',
                inputs=[transpose_name0],
                outputs=[softmax_name],
                axis=-1)
            transpose_name1 = self.get_name(op.type, 'transpose')
            transpose_node1 = helper.make_node(
                'Transpose',
                inputs=[softmax_name],
                outputs=op.output('Out'),
                perm=perm)
            return [transpose_node0, softmax_node, transpose_node1]

    def scale(self, op, block):
        scale = op.attr('scale')
        bias = op.attr('bias')
        if math.fabs(scale - 1.0) < 1e-06 and math.fabs(bias - 0.0) < 1e-06:
            node = helper.make_node(
                'Identity', inputs=op.input('X'), outputs=op.output('Out'))
            return node
        else:
            scale_name = self.get_name(op.type, 'scale')
            bias_name = self.get_name(op.type, 'bias')
            scale_node = self.make_constant_node(
                scale_name, onnx_pb.TensorProto.FLOAT, scale)
            bias_node = self.make_constant_node(bias_name,
                                                onnx_pb.TensorProto.FLOAT, bias)
            temp_tensor_name = self.get_name(op.type, 'temporary')
            if op.attr('bias_after_scale'):
                node1 = helper.make_node(
                    'Mul',
                    inputs=[scale_name, op.input('X')[0]],
                    outputs=[temp_tensor_name])
                node2 = helper.make_node(
                    'Add',
                    inputs=[bias_name, temp_tensor_name],
                    outputs=op.output('Out'))
            else:
                node1 = helper.make_node(
                    'Add',
                    inputs=[bias_name, op.input('X')[0]],
                    outputs=temp_tensor_name)
                node2 = helper.make_node(
                    'Mul',
                    inputs=[scale_name, temp_tensor_name],
                    outputs=[op.output('Out')])
            return [scale_node, bias_node, node1, node2]

    def mul(self, op, block):
        x_shape = block.var(op.input('X')[0]).shape
        y_shape = block.var(op.input('Y')[0]).shape
        out_shape = list(block.var(op.output('Out')[0]).shape)
        x_num_col_dims = op.attr('x_num_col_dims')
        y_num_col_dims = op.attr('y_num_col_dims')
        flatten_x_name = 'flatten_{}'.format(op.input('X')[0])
        flatten_y_name = 'flatten_{}'.format(op.input('Y')[0])
        shape_name = 'temp_shape_{}'.format(op.output('Out')[0])
        temp_out_name = 'temp_{}'.format(op.output('Out')[0])
        flatten_x = helper.make_node(
            'Flatten',
            inputs=op.input('X'),
            outputs=[flatten_x_name],
            axis=x_num_col_dims)
        flatten_y = helper.make_node(
            'Flatten',
            inputs=op.input('Y'),
            outputs=[flatten_y_name],
            axis=y_num_col_dims)
        shape_node = self.make_constant_node(
            shape_name, onnx_pb.TensorProto.INT64, out_shape)
        node = helper.make_node(
            'MatMul',
            inputs=[flatten_x_name, flatten_y_name],
            outputs=[temp_out_name])
        reshape_out = helper.make_node(
            'Reshape',
            inputs=[temp_out_name, shape_name],
            outputs=op.output('Out'))
        return [flatten_x, flatten_y, shape_node, node, reshape_out]

    def batch_norm(self, op, block):
        kwargs = {
            'epsilon': op.attr('epsilon'),
            'momentum': op.attr('momentum')
        }
        inputs = op.input('X') + op.input('Scale') + op.input(
            'Bias') + op.input('Mean') + op.input('Variance')
        node = helper.make_node(
            'BatchNormalization',
            inputs=inputs,
            outputs=op.output('Y'),
            **kwargs)
        return node

    def instance_norm(self, op, block):
        kwargs = {'epsilon': op.attr('epsilon'), }
        inputs = op.input('X') + op.input('Scale') + op.input('Bias')
        node = helper.make_node(
            'InstanceNormalization',
            inputs=inputs,
            outputs=op.output('Y'),
            **kwargs)
        return node

    def concat(self, op, block):
        node = helper.make_node(
            'Concat',
            inputs=op.input('X'),
            outputs=op.output('Out'),
            axis=op.attr('axis'))
        return node

    def sum(self, op, block):
        node = helper.make_node(
            'Sum', inputs=op.input('X'), outputs=op.output('Out'))
        return node

    def floor(self, op, block):
        node = helper.make_node(
            'Floor', inputs=op.input('X'), outputs=op.output('Out'))
        return node

    def uniform_random_batch_size_like(self, op, block):
        node = helper.make_node(
            'RandomUniformLike',
            inputs=op.input('Input'),
            outputs=op.output('Out'),
            high=op.attr('max'),
            dtype=self.paddle_onnx_dtype_map[op.attr('dtype')],
            low=op.attr('min'),
            seed=float(op.attr('seed')), )
        return node

    def depthwise_conv2d(self, op, block):
        return self.conv2d(op, block)

    def relu6(self, op, block):
        threshold = op.attr('threshold')
        node = helper.make_node(
            'Clip',
            inputs=[op.input('X')[0]],
            outputs=op.output('Out'),
            max=threshold,
            min=0.0)
        return [node]

    def shape(self, op, block):
        node = helper.make_node(
            'Shape', inputs=op.input('Input'), outputs=op.output('Out'))
        return node

    def split(self, op, block):
        sections = op.attr('sections')
        if len(sections) > 0:
            node = helper.make_node(
                'Split',
                inputs=op.input('X'),
                outputs=op.output('Out'),
                axis=op.attr('axis'),
                split=sections)
        else:
            node = helper.make_node(
                'Split',
                inputs=op.input('X'),
                outputs=op.output('Out'),
                axis=op.attr('axis'))
        return node

    def slice(self, op, block):
        axes = op.attr('axes')
        starts = op.attr('starts')
        ends = op.attr('ends')
        node = helper.make_node(
            "Slice",
            inputs=[op.input('Input')[0]],
            outputs=op.output('Out'),
            axes=axes,
            starts=starts,
            ends=ends)
        return [node]

    def fill_constant(self, op, block):
        value = op.attr('value')
        dtype = op.attr('dtype')
        shape = op.attr('shape')
        value = np.ones(shape) * value
        if dtype == 2:
            value = value.astype('int32')
        node = helper.make_node(
            'Constant',
            inputs=[],
            outputs=op.output('Out'),
            value=helper.make_tensor(
                name=op.output('Out')[0],
                data_type=self.paddle_onnx_dtype_map[dtype],
                dims=shape,
                vals=value.tolist()))
        return node

    def transpose2(self, op, block):
        node = helper.make_node(
            'Transpose',
            inputs=op.input('X'),
            outputs=op.output('Out'),
            perm=op.attr('axis'))
        return node

    def flatten2(self, op, block):
        node = helper.make_node(
            'Flatten',
            inputs=op.input('X'),
            outputs=op.output('Out'),
            axis=op.attr('axis'))
        return node

    def reshape2(self, op, block):
        input_names = op.input_names
        if len(op.input('ShapeTensor')) > 1:
            cast_shape_nodes = list()
            cast_shape_names = list()
            for i in range(len(op.input('ShapeTensor'))):
                dim = op.input('ShapeTensor')[i]
                temp_name = self.get_name(op.type, 'shape.cast')
                node = helper.make_node(
                    'Cast',
                    inputs=[dim],
                    outputs=[temp_name],
                    to=onnx_pb.TensorProto.INT64)
                cast_shape_nodes.append(node)
                cast_shape_names.append(temp_name)

            temp_name = self.get_name(op.type, 'shape.concat')
            shape_node = helper.make_node(
                'Concat', inputs=cast_shape_names, outputs=[temp_name], axis=-1)
            node = helper.make_node(
                'Reshape',
                inputs=[op.input('X')[0], temp_name],
                outputs=op.output('Out'))
            return cast_shape_nodes + [shape_node, node]
        elif len(op.input('ShapeTensor')) == 1:
            temp_name = self.get_name(op.type, 'shape.cast')
            cast_shape_node = helper.make_node(
                'Cast',
                inputs=op.input('ShapeTensor'),
                outputs=[temp_name],
                to=onnx_pb.TensorProto.INT64)
            node = helper.make_node(
                'Reshape',
                inputs=[op.input('X')[0], temp_name],
                outputs=op.output('Out'))
            return [cast_shape_node, node]
        elif op.attr('shape') is not None and len(op.attr('shape')) > 0:
            shape_name = self.get_name(op.type, 'shape')
            shape_node = self.make_constant_node(shape_name,
                                                 onnx_pb.TensorProto.INT64,
                                                 op.attr('shape'))
            reshape_node = helper.make_node(
                'Reshape',
                inputs=[op.input('X')[0], shape_name],
                outputs=op.output('Out'))
            return [shape_node, reshape_node]

    def dropout(self, op, block):
        dropout_mode = op.attr('dropout_implementation')
        dropout_prob = op.attr('dropout_prob')
        if dropout_mode == 'upscale_in_train':
            node = helper.make_node(
                'Identity', inputs=op.input('X'), outputs=op.output('Out'))
            return node
        elif dropout_mode == 'downgrade_in_infer':
            scale_name = self.get_name(op.type, 'scale')
            scale_node = self.make_constant_node(
                scale_name, onnx_pb.TensorProto.FLOAT, 1 - dropout_prob)
            node = helper.make_node(
                "Mul",
                inputs=[op.input('X')[0], scale_name],
                outputs=op.output('Out'))
            return [scale_node, node]
        else:
            raise Exception("Unexpected situation happend")

    def reduce_mean(self, op, block):
        node = helper.make_node(
            'ReduceMean',
            inputs=op.input('X'),
            outputs=op.output('Out'),
            axes=op.attr('dim'),
            keepdims=op.attr('keep_dim'))
        return node

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

    def hard_sigmoid(self, op, block):
        slope = op.attr('slope')
        offset = op.attr('offset')
        node = helper.make_node(
            'HardSigmoid',
            inputs=op.input('X'),
            outputs=op.output('Out'),
            alpha=slope,
            beta=offset)
        return node

    def swish(self, op, block):
        beta = op.attr('beta')
        beta_name = self.get_name(op.type, 'beta')
        beta_node = onnx.helper.make_node(
            'Constant',
            name=beta_name,
            inputs=[],
            outputs=[beta_name],
            value=onnx.helper.make_tensor(
                name=beta_name,
                data_type=onnx.TensorProto.FLOAT,
                dims=(),
                vals=[beta]))

        beta_x_name = self.get_name(op.type, 'beta_x')
        beta_x_node = onnx.helper.make_node(
            'Mul',
            name=beta_x_name,
            inputs=[op.input('X')[0], beta_name],
            outputs=[beta_x_name])
        sigmoid_name = self.get_name(op.type, 'sigmoid')
        sigmoid_node = onnx.helper.make_node(
            'Sigmoid',
            name=sigmoid_name,
            inputs=[beta_x_name],
            outputs=[sigmoid_name])
        swish_node = onnx.helper.make_node(
            'Mul',
            inputs=[op.input('X')[0], sigmoid_name],
            outputs=op.output('Out'))
        return [beta_node, beta_x_node, sigmoid_node, swish_node]

    def hard_swish(self, op, block):
        scale_name = self.get_name(op.type, 'scale')
        offset_name = self.get_name(op.type, 'offset')
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
        min_value = 0.0
        max_value = op.attr('threshold')
        node1 = helper.make_node(
            'Clip',
            inputs=[name0],
            outputs=[name1],
            max=max_value,
            min=min_value)
        name2 = self.get_name(op.type, 'mul')
        node2 = helper.make_node(
            'Mul', inputs=[op.input('X')[0], name1], outputs=[name2])
        node3 = helper.make_node(
            'Div', inputs=[name2, scale_name], outputs=op.output('Out'))
        return [scale_node, offset_node, node0, node1, node2, node3]

    def elementwise_mul(self, op, block):
        axis = op.attr('axis')
        x_shape = block.var(op.input('X')[0]).shape
        y_shape = block.var(op.input('Y')[0]).shape
        if len(y_shape) == 1 and axis == 1:
            shape_name = self.get_name(op.type, 'shape')
            shape_value = [1] * len(x_shape)
            shape_value[axis] = y_shape[0]
            shape_node = self.make_constant_node(
                shape_name, onnx_pb.TensorProto.INT64, shape_value)
            temp_value = self.get_name(op.type, 'temp')
            y_node = helper.make_node(
                'Reshape',
                inputs=[op.input('Y')[0], shape_name],
                outputs=[temp_value])
            node = helper.make_node(
                'Mul',
                inputs=[op.input('X')[0], temp_value],
                outputs=op.output('Out'))
            return [shape_node, y_node, node]
        elif axis == -1 or axis == (len(x_shape) - 1
                                    ) or len(x_shape) == len(y_shape):
            node = helper.make_node(
                'Mul',
                inputs=[op.input('X')[0], op.input('Y')[0]],
                outputs=op.output('Out'))
            return node
        else:
            raise Exception("Unexpected situation happend in elementwise_mul")
        return node

    def feed(self, op, block):
        name = op.output('Out')[0]
        var = block.var(name)
        tensor_info = helper.make_tensor_value_info(
            name=name,
            shape=var.shape,
            elem_type=self.paddle_onnx_dtype_map[var.dtype])
        return tensor_info

    def fetch(self, op, block):
        name = op.input('X')[0]
        var = block.var(name)
        tensor_info = helper.make_tensor_value_info(
            name=name,
            shape=var.shape,
            elem_type=self.paddle_onnx_dtype_map[var.dtype])
        return tensor_info

    def unsqueeze2(self, op, block):
        node = helper.make_node(
            'Unsqueeze',
            inputs=op.input('X'),
            outputs=op.output('Out'),
            axes=op.attr('axes'))
        return node

    def cast(self, op, block):
        node = helper.make_node(
            'Cast',
            inputs=op.input('X'),
            outputs=op.output('Out'),
            to=self.paddle_onnx_dtype_map[op.attr('out_dtype')])
        return node

    def arg_max(self, op, block):
        node = helper.make_node(
            'ArgMax',
            inputs=op.input('X'),
            outputs=op.output('Out'),
            axis=op.attr('axis'),
            keepdims=0)
        return node

    def reciprocal(self, op, block):
        inputs = op.input(op.input_names[0])
        outputs = op.output(op.output_names[0])
        node = helper.make_node('Reciprocal', inputs=inputs, outputs=outputs)
        return node

    def im2sequence(self, op, block):
        from .paddle_custom_layer.im2sequence import im2sequence
        return im2sequence(op, block)

    def yolo_box(self, op, block):
        from .paddle_custom_layer.yolo_box import yolo_box
        return yolo_box(op, block)

    def multiclass_nms(self, op, block):
        from .paddle_custom_layer.multiclass_nms import multiclass_nms
        return multiclass_nms(op, block)

    def box_coder(self, op, block):
        from .paddle_custom_layer.box_coder import box_coder
        return box_coder(op, block)

    def prior_box(self, op, block):
        from .paddle_custom_layer.prior_box import prior_box
        return prior_box(op, block)
