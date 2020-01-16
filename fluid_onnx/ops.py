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
import onnx
import numpy as np
import re
from collections import namedtuple
from functools import partial
from onnx import TensorProto
from onnx.numpy_helper import from_array
from onnx.helper import make_node, make_tensor
from paddle.fluid.executor import _fetch_var as fetch_var
from fluid.utils import op_io_info, get_old_name
from fluid_onnx.variables import PADDLE_TO_ONNX_DTYPE, PADDLE_DTYPE_DICT, paddle_onnx_shape
from fluid_onnx.detection_ops import *
"""
Priority of ops (uniques) to figure out support for.

test_fit_a_line.py
- mean
- mul
- elementwise_add
- elementwise_sub
- fill_constant

^ Try to make this run before proceeding.

test_machine_translation.py
- tanh
- lstm
- sequence_pool
- lod_rank_table
- max_sequence_len
- less_than
- lod_tensor_to_array
- write_to_array
- while
- array_to_lod_tensor
- cross_entropy
- lod_tensor_to_array
- read_from_array
- sum
- scale
- adagrad
- shrink_rnn_memory
- softmax
- write_to_array
- increment
"""

__onnx_ver__ = onnx.version.version

__name_prefix__ = ""
_prev_program_id_ = None
_global_constants_ = dict()
_unique_name_index_ = dict()
_ConstantNode = namedtuple('_ConstantNode', ['name', 'node'])
_invalid_name_re = re.compile(r'[^A-Za-z0-9_./>]')


def _check_and_reset_block_name_space(block):
    global _prev_program_id_
    # maybe some better way to do this?
    if id(block.program) != _prev_program_id_:
        _global_constants_.clear()
        _unique_name_index_.clear()
        _prev_program_id_ = id(block.program)


def _tmp_name(name, suffix, block):
    """
    Create a unique temporary tensor name in the form of `{name}/{suffix}{index}` where `index` is blank if the
    `{name}/{suffix}` is used for the first time, else the index is the number of times the base name has been created.
    """
    _check_and_reset_block_name_space(block)
    suffix = suffix.replace('/', '')
    name = name + '_' + suffix
    current = _unique_name_index_.get(name, 0)
    _unique_name_index_[name] = (current + 1)
    return name if current == 0 else name + str(current)


def _to_node_tuple(node_or_list):
    if isinstance(node_or_list, tuple):
        node_or_list = list(node_or_list)
    if not isinstance(node_or_list, list):
        node_or_list = [node_or_list]
    return tuple(node for node in node_or_list if node is not None)


def global_const(value, block, node_name=None):
    _check_and_reset_block_name_space(block)
    np_val = np.array(value)
    if len(np_val.shape) != 0 and node_name is None:
        raise KeyError('value must be a scala or the node_name must be given. it has a shape of ' + str(np_val.shape))
    if node_name is None:
        node_name = 'C_' + str(value)
    node_name = _invalid_name_re.sub('_', node_name)
    node = _global_constants_.get(node_name, None)
    if node:
        return _ConstantNode(node_name, None)
    else:
        node = onnx.helper.make_node(
            'Constant',
            inputs=[],
            outputs=[node_name],
            name=node_name,
            value=from_array(np_val, name=node_name + '_const_tensor'))
        _global_constants_[node_name] = node
        return _ConstantNode(node_name, node)


def activation_ops(act_type, operator, block):
    """ Convert common activations with type specified by 'act_type', including
        'abs', 'ceil', 'exp', 'floor', 'log', 'reciprocal', 'relu', 'sigmoid',
        'softplus', 'softsign', 'sqrt' and 'tanh'.
    """

    inputs, _, outputs = op_io_info(operator)
    return make_node(
        act_type,
        inputs=list(inputs.values())[0],
        outputs=list(outputs.values())[0])


def arg_max_op(operator, block):
    inputs, attrs, outputs = op_io_info(operator)
    node_list = []
    axis = attrs['axis']
    outputs_argmax = [outputs['Out'][0] + '@argmax']
    argmax_node = make_node(
        'ArgMax',
        inputs=inputs['X'],
        outputs=outputs_argmax,
        axis=axis,
        keepdims=0)
    node_list.append(argmax_node)

    cast_node = make_node(
        'Cast',
        inputs=outputs_argmax,
        outputs=outputs['Out'],
        to=1)

    node_list.append(cast_node)
    return tuple(node_list)


def argmin_op():
    pass


def batch_norm_op(operator, block):
    inputs, attrs, outputs = op_io_info(operator)

    x_shape = block.vars[get_old_name(inputs['X'][0])].shape
    nodes = ()
    # Batch norm in ONNX only supports input dim. >= 3, for input tensor with
    # dim. == 2 supported by Fluid, reshape it into dim. == 4.
    if len(x_shape) == 2:
        new_shape = [0, x_shape[1], 1, 1]
        reshaped_x = [inputs['X'][0] + '@reshape_0']
        if __onnx_ver__ == '1.0.1':
            reshape_node = make_node(
                'Reshape',
                inputs=inputs['X'],
                shape=new_shape,
                outputs=reshaped_x)
        else:
            new_shape_name = [inputs['X'][0] + '@shape_tensor_0']
            new_shape_node = make_node(
                'Constant',
                inputs=[],
                outputs=new_shape_name,
                value=make_tensor(
                    name=new_shape_name[0],
                    data_type=TensorProto.INT64,
                    dims=(4, ),
                    vals=new_shape))
            reshape_node = make_node(
                'Reshape',
                inputs=inputs['X'] + new_shape_name,
                outputs=reshaped_x)
            nodes = (new_shape_node, )
        nodes += (reshape_node, )
    else:
        reshaped_x = inputs['X']

    kwargs = {'epsilon': attrs['epsilon'], 'momentum': attrs['momentum']}
    bn_node = make_node(
        'BatchNormalization',
        inputs=reshaped_x + inputs['Scale'] + inputs['Bias'] + inputs['Mean'] +
        inputs['Variance'],
        outputs=outputs['Y'],
        **kwargs)

    return nodes + (bn_node, )


def bilinear_interp_op(operator, block):
    inputs, attrs, outputs = op_io_info(operator)
    input_shape = block.vars[get_old_name(inputs['X'][0])].shape
    batch_size = input_shape[0]
    channels = input_shape[1]
    height = input_shape[2]
    width = input_shape[3]
    node_list = []
    im_outputs = []

    if inputs['OutSize'] == []:
        out_h_w = [attrs['out_h'], attrs['out_w']]
        name_out_h_w = [outputs['Out'][0] + "@out_h_w"]
        node_out_h_w = make_node(
            'Constant',
            inputs=[],
            outputs=name_out_h_w,
            value=make_tensor(
                name=name_out_h_w[0],
                data_type=TensorProto.FLOAT,
                dims=[2],
                vals=out_h_w))
        node_list.append(node_out_h_w)

        outputs_out_size_f = [outputs['Out'][0] + "@out_size_f"]
        node_out_size_f = make_node(
            'Cast', inputs=name_out_h_w, outputs=outputs_out_size_f, to=1)
        node_list.append(node_out_size_f)
    else:
        outputs_out_size_f = [outputs['Out'][0] + "@out_size_f"]
        node_out_size_f = make_node(
            'Cast', inputs=inputs['OutSize'], outputs=outputs_out_size_f, to=1)
        node_list.append(node_out_size_f)

    name_h_w = [outputs['Out'][0] + "@h_w"]
    node_h_w = make_node(
        'Constant',
        inputs=[],
        outputs=name_h_w,
        value=make_tensor(
            name=name_h_w[0],
            data_type=TensorProto.FLOAT,
            dims=[2],
            vals=[height, width]))
    node_list.append(node_h_w)

    outputs_h_w_scales = [outputs['Out'][0] + "@h_w_scales"]
    node_h_w_scales = make_node(
        'Div', inputs=outputs_out_size_f + name_h_w, outputs=outputs_h_w_scales)
    node_list.append(node_h_w_scales)

    name_b_c_scales = [outputs['Out'][0] + "@b_c_scales"]
    node_b_c_scales = make_node(
        'Constant',
        inputs=[],
        outputs=name_b_c_scales,
        value=make_tensor(
            name=name_b_c_scales[0],
            data_type=TensorProto.FLOAT,
            dims=[2],
            vals=[1, 1]))
    node_list.append(node_b_c_scales)

    outputs_scales = [outputs['Out'][0] + "@scales"]

    node_scales = make_node(
        'Concat',
        inputs=name_b_c_scales + outputs_h_w_scales,
        outputs=outputs_scales,
        axis=0)
    node_list.append(node_scales)

    outputs_resize = [outputs['Out'][0] + "@resize"]
    node_resize = make_node(
        'Resize',
        inputs=inputs['X'] + outputs_scales,
        outputs=outputs['Out'],
        mode='linear')
    node_list.append(node_resize)
    return tuple(node_list)


def cast_op(operator, block):
    inputs, attrs, outputs = op_io_info(operator)
    return make_node(
        'Cast',
        inputs=inputs['X'],
        outputs=outputs['Out'],
        to=PADDLE_TO_ONNX_DTYPE[attrs['out_dtype']])


def clip_op(operator, block):
    inputs, attrs, outputs = op_io_info(operator)
    return make_node(
        'Clip',
        inputs=inputs['X'],
        outputs=outputs['Out'],
        min=attrs['min'],
        max=attrs['max'])


def compare_ops(op_type, operator, block):
    ''' Conversion for compare ops, including 'Less', 'Equal', 'Greater'
    '''
    inputs, attrs, outputs = op_io_info(operator)
    return make_node(
        op_type, inputs=inputs['X'] + inputs['Y'], outputs=outputs['Out'])


def concat_op(operator, block):
    inputs, attrs, outputs = op_io_info(operator)
    return make_node(
        'Concat',
        inputs=inputs['X'],
        outputs=outputs['Out'],
        axis=attrs['axis'])


def constant_op(var, scope):
    data = fetch_var(var.name, scope)
    constant_node = make_node(
        'Constant',
        inputs=[],
        outputs=[var.name],
        value=make_tensor(
            name=var.name,
            dims=var.shape,
            data_type=PADDLE_TO_ONNX_DTYPE[var.dtype],
            vals=data.flatten().tolist()))
    return constant_node


def conv2d_op(operator, block):
    inputs, attrs, outputs = op_io_info(operator)

    kernel_shape = block.vars[get_old_name(inputs['Filter'][0])].shape
    conv2d = make_node(
        'Conv',
        inputs=inputs['Input'] + inputs['Filter'],
        outputs=outputs['Output'],
        dilations=attrs['dilations'],
        kernel_shape=kernel_shape[-2:],
        strides=attrs['strides'],
        group=attrs['groups'],
        pads=attrs['paddings'] + attrs['paddings'])
    return conv2d


def conv2d_transpose_op(operator, block):
    inputs, attrs, outputs = op_io_info(operator)

    kernel_shape = block.vars[get_old_name(inputs['Filter'][0])].shape
    conv2d_transpose = make_node(
        'ConvTranspose',
        inputs=inputs['Input'] + inputs['Filter'],
        outputs=outputs['Output'],
        dilations=attrs['dilations'],
        kernel_shape=kernel_shape[-2:],
        strides=attrs['strides'],
        group=1,
        pads=attrs['paddings'] + attrs['paddings'])
    return conv2d_transpose


def depthtospace_op():
    pass


def _scale_op(input, output, scale):
    scale_const_node = global_const(scale)
    return [
        scale_const_node.node,
        make_node(
            'Mul',
            inputs=[input, scale_const_node.name],
            outputs=[output])
    ]

def dropout_op(operator, block):
    inputs, attrs, outputs = op_io_info(operator)
    scale_input = [outputs['Out'][0] + '@dropout']
    dropout_node = make_node(
        'Dropout',
        inputs=inputs['X'],
        outputs=scale_input + outputs['Mask'],
        ratio=attrs['dropout_prob'])

    ## Fluid and ONNX use different dropout formula
    # onnx version >= 1.4.1 supports ai.onnx ver 9 which has no `broadcast` attr for Mul
    scale_mul_attrs = {'broadcast': 1} if __onnx_ver__ < '1.4.1' else {}
    scale_const_node = global_const(np.array(1.0 - attrs['dropout_prob'], dtype=np.float32), block)
    return _to_node_tuple([
        dropout_node,
        scale_const_node.node,
        make_node(
            'Mul',
            inputs=scale_input + [scale_const_node.name],
            outputs=outputs['Out'],
            **scale_mul_attrs)])


def elementwise_ops(op_type, operator, block):
    """Convert elementwise operators From to ONNX. Supported elementwise
       'op_type' includes 'Add', 'Div', 'Mul', 'Pow' and 'Sub'.
    """

    inputs, attrs, outputs = op_io_info(operator)
    node_list = []
    Y_shape_name = inputs['Y']
    if 'axis' in attrs and attrs['axis'] != -1:
        shape_x = block.vars[get_old_name(inputs['X'][0])].shape
        shape_y = block.vars[get_old_name(inputs['Y'][0])].shape
        rank_x = len(shape_x)
        rank_y = len(shape_y)
        axis = rank_x - rank_y if attrs['axis'] == -1 else attrs['axis']
        shape = list(shape_y)
        pre_shape = []
        post_shape = []
        if axis > 0:
            for i in range(0, axis):
                pre_shape.append(1)
        if axis + len(shape) < rank_x:
            for i in range(axis + len(shape), rank_x):
                post_shape.append(1)
        pre_shape.extend(shape)
        pre_shape.extend(post_shape)
        final_shape = [i if i > 0 else 1 for i in pre_shape]
        shape_name = _tmp_name(outputs['Out'][0], "@shape_var", block)
        output_const_node = make_node(
            'Constant',
            inputs=[],
            outputs=[shape_name],
            value=make_tensor(
                name=shape_name + "@const",
                data_type=TensorProto.INT64,
                dims=[len(final_shape)],
                vals=final_shape))

        output_shape_name = [outputs['Out'][0] + "@reshape_y"]
        output_shape_node = make_node(
            'Reshape',
            inputs=[inputs['Y'][0], shape_name],
            outputs=output_shape_name)
        node_list.extend([output_const_node, output_shape_node])
        Y_shape_name = output_shape_name
    node_type = make_node(
        op_type, inputs=inputs['X'] + Y_shape_name, outputs=outputs['Out'])
    node_list.append(node_type)
    return tuple(node_list)


def elu_op(operator, block):
    inputs, attrs, outputs = op_io_info(operator)
    return make_node(
        'Elu', inputs=inputs['X'], outputs=outputs['Out'], alpha=attrs['alpha'])


def equal_op():
    pass


def flatten_op(operator, block):
    inputs, attrs, outputs = op_io_info(operator)
    axis = attrs['axis']
    return make_node(
        'Flatten', inputs=inputs['X'], outputs=outputs['Out'], axis=axis)


def fill_constant_op(operator, block):
    inputs, attrs, outputs = op_io_info(operator)
    shape = attrs['shape']
    dtype = attrs['dtype']
    value = attrs['value']
    mat = np.ones(np.array(shape)) * value
    if dtype == 5:
        dtype = 'float32'
        mat = list(mat.reshape(-1).astype('float32'))
    elif dtype == 2:
        dtype = 'int32'
        mat = list(mat.reshape(-1).astype('int32'))
    value_dtype = PADDLE_DTYPE_DICT[dtype]
    node_list = []
    outputs_fill_constant_value = [outputs['Out'][0] + "@fill_constant_value"]
    node_fill_constant_value = make_node(
        'Constant',
        inputs=[],
        outputs=outputs_fill_constant_value,
        value=make_tensor(
            name=outputs_fill_constant_value[0],
            dims=[len(mat)],
            data_type=PADDLE_TO_ONNX_DTYPE[value_dtype],
            vals=mat))
    node_list.append(node_fill_constant_value)

    outputs_value_shape = [outputs['Out'][0] + "@value_shape"]
    node_value_shape = make_node(
        'Constant',
        inputs=[],
        outputs=outputs_value_shape,
        value=make_tensor(
            name=outputs_value_shape[0],
            dims=[len(shape)],
            data_type=TensorProto.INT64,
            vals=shape))
    node_list.append(node_value_shape)

    node_constant_reshape = make_node(
        'Reshape',
        inputs=outputs_fill_constant_value + outputs_value_shape,
        outputs=outputs['Out'])
    node_list.append(node_constant_reshape)
    return tuple(node_list)


def gru_op():
    pass


def gather_op(operator, block):
    inputs, attrs, outputs = op_io_info(operator)
    return make_node(
        'Gather', inputs=inputs['X'] + inputs['Index'], outputs=outputs['Out'])


def gelu_op(operator, block):
    inputs, attrs, outputs = op_io_info(operator)
    name_x_div_sqrt2 = _tmp_name(inputs['X'][0], 'div_sqrt2', block)
    name_x_erf = _tmp_name(inputs['X'][0], 'erf', block)
    name_erf_plus_one = outputs['Out'][0] + '@erf_plus'
    name_x_mul_erf = outputs['Out'][0] + '@x_mul_erf'

    sqrt2 = np.sqrt(np.array(2.0, dtype=np.float32))
    sqrt2_node = global_const(sqrt2, block)
    one_node = global_const(np.array(1.0, dtype=np.float32), block)
    two_node = global_const(np.array(2.0, dtype=np.float32), block)
    node_list = [node.node for node in [sqrt2_node, one_node, two_node] if node.node]
    node_list.extend([
        make_node(
            'Div',
            inputs=[inputs['X'][0], sqrt2_node.name],
            outputs=[name_x_div_sqrt2]),
        make_node(
            'Erf',
            inputs=[name_x_div_sqrt2],
            outputs=[name_x_erf]),
        make_node(
            'Add',
            inputs=[name_x_erf, one_node.name],
            outputs=[name_erf_plus_one]),
        make_node(
            'Mul',
            inputs=[inputs['X'][0], name_erf_plus_one],
            outputs=[name_x_mul_erf]),
        make_node(
            'Div',
            inputs=[name_x_mul_erf, two_node.name],
            outputs=outputs['Out'])
    ])
    return tuple(node_list)


def gemm_op():
    pass


def globallppool_op():
    pass


def hardsigmoid_op(operator, block):
    inputs, attrs, outputs = op_io_info(operator)
    return make_node(
        'HardSigmoid',
        inputs=inputs['X'],
        outputs=outputs['Out'],
        alpha=0.2,
        beta=0.5)
    pass


def hardmax_op():
    pass


def instancenormalization_op():
    pass


def layer_norm_op(operator, block):
    inputs, attrs, outputs = op_io_info(operator)
    name_x = inputs['X'][0]
    name_bias = inputs['Bias'][0]
    name_scale = inputs['Scale'][0]
    name_mean = outputs['Mean'][0]
    name_variance = outputs['Variance'][0]
    name_y = outputs['Y'][0]
    norm_axis = attrs['begin_norm_axis']
    norm_epsilon = np.array(attrs['epsilon'], dtype=np.float32)
    name_x_sub_mean = _tmp_name(name_x, 'sub_mean', block)
    name_x_sub_mean_sq = _tmp_name(name_x, 'sub_mean_sq', block)
    name_x_stddev = _tmp_name(name_x, 'stddev', block)
    name_x_norm = _tmp_name(name_x, 'norm', block)
    name_x_norm_scaled = _tmp_name(name_x, 'norm_scaled', block)
    name_mean_unflatten = _tmp_name(name_mean, 'unflatten', block)
    name_variance_unflatten = _tmp_name(name_variance, 'unflatten', block)
    name_mean_flatten2d = _tmp_name(name_mean, 'flatten2d', block)
    name_variance_flatten2d = _tmp_name(name_variance, 'flatten2d', block)
    epsilon_node = global_const(norm_epsilon, block)
    node_list = []
    if epsilon_node.node:
        node_list.append(epsilon_node.node)
    node_list.extend([
        onnx.helper.make_node(
            'ReduceMean',
            inputs=[name_x],
            outputs=[name_mean_unflatten],
            axes=[norm_axis],
            keepdims=1),
        onnx.helper.make_node(
            'Sub',
            inputs=[name_x, name_mean_unflatten],
            outputs=[name_x_sub_mean],
        ),
        onnx.helper.make_node(
            'Mul',
            inputs=[name_x_sub_mean, name_x_sub_mean],
            outputs=[name_x_sub_mean_sq],
        ),
        onnx.helper.make_node(
            'ReduceMean',
            inputs=[name_x_sub_mean_sq],
            outputs=[name_variance_unflatten],
            axes=[norm_axis],
            keepdims=1),
        onnx.helper.make_node(
            'Add',
            inputs=[name_variance_unflatten, epsilon_node.name],
            outputs=[(name_variance + '@add_epsilon')]),
        onnx.helper.make_node(
            'Sqrt',
            inputs=[(name_variance + '@add_epsilon')],
            outputs=[name_x_stddev]),
        onnx.helper.make_node(
            'Div',
            inputs=[name_x_sub_mean, name_x_stddev],
            outputs=[name_x_norm]),
        onnx.helper.make_node(
            'Mul',
            inputs=[name_x_norm, name_scale],
            outputs=[name_x_norm_scaled]),
        onnx.helper.make_node(
            'Add',
            inputs=[name_x_norm_scaled, name_bias],
            outputs=[name_y]),
        onnx.helper.make_node(
            'Flatten',
            inputs=[name_mean_unflatten],
            outputs=[name_mean_flatten2d],
            axis=0),
        onnx.helper.make_node(
            'Flatten',
            inputs=[name_variance_unflatten],
            outputs=[name_variance_flatten2d],
            axis=0),
        onnx.helper.make_node(
            'Squeeze',
            inputs=[name_mean_flatten2d],
            outputs=[name_mean],
            axes=[0]),
        onnx.helper.make_node(
            'Squeeze',
            inputs=[name_variance_flatten2d],
            outputs=[name_variance],
            axes=[0]),
    ])
    return _to_node_tuple(node_list)


def lrn_op(operator, block):
    inputs, attrs, outputs = op_io_info(operator)
    return make_node(
        'LRN',
        inputs=inputs['X'],
        outputs=outputs['Out'],
        alpha=attrs['alpha'],
        beta=attrs['beta'],
        bias=attrs['k'],
        size=attrs['n'])


def lstm_op():
    pass


def leaky_relu_op(operator, block):
    inputs, attrs, outputs = op_io_info(operator)
    return make_node(
        'LeakyRelu',
        inputs=inputs['X'],
        outputs=outputs['Out'],
        alpha=attrs['alpha'])


def binary_logical_ops(op_type, operator, block):
    """Convert binary logical operators, i.e. 'And', 'Or' and 'Xor'.
    """

    inputs, _, outputs = op_io_info(operator)
    return make_node(
        op_type, inputs=inputs['X'] + inputs['Y'], outputs=outputs['Out'])


def unary_logical_ops(op_type, operator, block):
    """Convert unary logical operators, i.e. 'Not'.
    """

    inputs, _, outputs = op_io_info(operator)
    return make_node(op_type, inputs=inputs['X'], outputs=outputs['Out'])


def logsoftmax_op():
    pass


def lookup_table_op(operator, block):
    inputs, attrs, outputs = op_io_info(operator)
    ids_shape = block.vars[get_old_name(inputs['Ids'][0])].shape
    W_shape = block.vars[get_old_name(inputs['W'][0])].shape
    out_shape = ids_shape[:-1] + W_shape[-1:]

    name_out_gathered = _tmp_name(outputs['Out'][0], 'gathered', block)

    node_list = []
    out_shape_node = global_const(np.array(out_shape), block,
                                  node_name='shape_'+'x'.join(str(d) if d > 0 else 'N' for d in out_shape))
    node_list.extend([
        out_shape_node.node,
        make_node(
            'Gather',
            inputs=[inputs['W'][0], inputs['Ids'][0]],
            outputs=[name_out_gathered],
            axis=0),
        make_node(
            'Reshape',
            inputs=[name_out_gathered, out_shape_node.name],
            outputs=outputs['Out'])
    ])
    return _to_node_tuple(node_list)


def lpnormalization_op():
    pass


def lppool_op():
    pass


def mul_op(operator, block):
    inputs, attrs, outputs = op_io_info(operator)

    # Get shape of inputs
    x_shape = block.vars[get_old_name(inputs['X'][0])].shape
    y_shape = block.vars[get_old_name(inputs['Y'][0])].shape
    x_num_col_dims, y_num_col_dims = attrs['x_num_col_dims'], attrs[
        'y_num_col_dims']
    out_shape = x_shape[:x_num_col_dims] + y_shape[y_num_col_dims:]

    # Flatten input(X) and input(Y) into 2-D matries
    x_flat_out = [_tmp_name(inputs['X'][0], 'flatten', block)]
    y_flat_out = [inputs['Y'][0] + '@flatten_0']

    # Because in TensorRT backend, Flatten op only accepts input tensor with
    # dimension 3, here we use Reshape op to flatten the input tensor when
    # ONNX is v1.0.1.
    if __onnx_ver__ == '1.0.1':
        # In v1.0.1, shape is the attribute of Reshape op, not an input tensor.
        x_shape = paddle_onnx_shape(x_shape)
        y_shape = paddle_onnx_shape(y_shape)
        flatten_x_node = make_node(
            'Reshape',
            inputs=inputs['X'],
            outputs=x_flat_out,
            shape=[
                np.prod(x_shape[:x_num_col_dims]),
                np.prod(x_shape[x_num_col_dims:])
            ])
        flatten_y_node = make_node(
            'Reshape',
            inputs=inputs['Y'],
            outputs=y_flat_out,
            shape=[
                np.prod(y_shape[:y_num_col_dims]),
                np.prod(y_shape[y_num_col_dims:])
            ])
    else:
        flatten_x_node = make_node(
            'Flatten',
            inputs=inputs['X'],
            outputs=x_flat_out,
            axis=attrs['x_num_col_dims'])
        flatten_y_node = make_node(
            'Flatten',
            inputs=inputs['Y'],
            outputs=y_flat_out,
            axis=attrs['y_num_col_dims'])

    # Mat mul
    matmul_out = [outputs['Out'][0] + '@matmul_0']
    matmul_node = make_node(
        'MatMul', inputs=x_flat_out + y_flat_out, outputs=matmul_out)

    nodes = (flatten_x_node, flatten_y_node, matmul_node)
    # Reshpe output
    if __onnx_ver__ == '1.0.1':
        output_node = make_node(
            'Reshape',
            inputs=matmul_out,
            shape=out_shape,
            outputs=outputs['Out'])
        nodes += (output_node, )
    else:
        output_shape_name = [outputs['Out'][0] + '@shape_0']
        output_shape_node = make_node(
            'Constant',
            inputs=[],
            outputs=output_shape_name,
            value=make_tensor(
                name=output_shape_name[0],
                data_type=TensorProto.INT64,
                dims=(len(out_shape), ),
                vals=out_shape))
        output_node = make_node(
            'Reshape',
            inputs=matmul_out + output_shape_name,
            outputs=outputs['Out'])
        nodes += (output_shape_node, output_node)

    return nodes


def max_op():
    pass


def maxroipool_op():
    pass


def mean_op():
    pass


def min_op():
    pass

def matmul_op(operator, block):
    inputs, attrs, outputs = op_io_info(operator)
    x_shape = block.vars[get_old_name(inputs['X'][0])].shape
    y_shape = block.vars[get_old_name(inputs['Y'][0])].shape
    name_matmul_x = inputs['X'][0]
    name_matmul_y = inputs['Y'][0]
    transposed = False
    node_list = []
    if attrs.get('transpose_X', False):
        transposed = True
        name_matmul_x, transposed_node = _matmul_transpose_op(name_matmul_x, block)
        node_list.append(transposed_node)
    if attrs.get('transpose_Y', False):
        transposed = True
        name_matmul_y, transposed_node = _matmul_transpose_op(name_matmul_y, block)
        node_list.append(transposed_node)
    name_matmul_out = outputs['Out'][0]
    name_mul_out = name_matmul_out
    need_unsqueeze = len(x_shape) == 1 and len(y_shape) == 1
    need_to_scale = attrs.get('alpha', None) != 1.0
    unsqueeze_inputs = []
    if need_to_scale or need_unsqueeze:
        name_matmul_out = _tmp_name(name_matmul_out, 'matmul', block)
    if need_to_scale and need_unsqueeze:
        name_mul_out = name_matmul_out + '_scaled'
        unsqueeze_inputs = [name_mul_out]
    elif not need_to_scale and need_unsqueeze:
        name_mul_out = _tmp_name(outputs['Out'][0], 'scaled', block)
        unsqueeze_inputs = [name_matmul_out]
    node_list.append(make_node(
        'MatMul',
        inputs=[name_matmul_x, name_matmul_y],
        outputs=[name_matmul_out]))
    if need_to_scale:
        scale_const_node = global_const(np.array(attrs['alpha'], dtype=np.float32), block)
        node_list.extend([
            scale_const_node.node,
            make_node(
                'Mul',
                inputs=[name_matmul_out, scale_const_node.name],
                outputs=[name_mul_out])
        ])
    if need_unsqueeze and not transposed:
        # According to: Also note that if the raw tensor x or y is rank-1 and nontransposed, the prepended or appended
        # dimension 1 will be removed after matrix multiplication.
        rank1_shape_const = global_const(np.array([-1], dtype=np.int64), block, node_name='rank1_shape')
        node_list.append(rank1_shape_const.node)
        node_list.append(onnx.helper.make_node(
            'Reshape',
            inputs=unsqueeze_inputs + [rank1_shape_const.name],
            outputs=outputs['Out']))
    return _to_node_tuple(node_list)


def _matmul_transpose_op(x, block):
    x_shape = block.vars[get_old_name(x)].shape
    name_x_transposed = _tmp_name(x, 'transposed', block)
    axes = np.arange(len(x_shape))
    axes = list(axes[:-2]) + [axes[-1], axes[-2]]
    node = make_node(
        'Transpose',
        inputs=[x],
        outputs=[name_x_transposed],
        perm=axes)
    return name_x_transposed, node


def neg_op():
    pass


def norm_op(operator, block):
    inputs, attrs, outputs = op_io_info(operator)
    return make_node(
        'LpNormalization',
        inputs=inputs['X'], outputs=outputs['Out'], axis=attrs['axis'])


def prelu_op(operator, block):
    inputs, attrs, outputs = op_io_info(operator)
    return make_node(
        'PRelu', inputs=inputs['X'] + inputs['Alpha'], outputs=outputs['Out'])


def pad_op():
    pass


def pool2d_op(operator, block):
    inputs, attrs, outputs = op_io_info(operator)
    if attrs['global_pooling'] is False:
        op_type = {'max': 'MaxPool', 'avg': 'AveragePool'}
        pool2d = make_node(
            op_type[attrs['pooling_type']],
            inputs=inputs['X'],
            outputs=outputs['Out'],
            kernel_shape=attrs['ksize'],
            strides=attrs['strides'],
            pads=attrs['paddings'] + attrs['paddings'], )
    else:
        op_type = {'max': 'GlobalMaxPool', 'avg': 'GlobalAveragePool'}
        pool2d = make_node(
            op_type[attrs['pooling_type']],
            inputs=inputs['X'],
            outputs=outputs['Out'])
    return pool2d


def pow_op():
    pass


def rnn_op():
    pass


def randomnormal_op():
    pass


def randomnormallike_op():
    pass


def randomuniform_op():
    pass


def randomuniformlike_op():
    pass


def reduce_ops(op_type, operator, block):
    """Convert reduce operators in Fluid to ONNX. 'op_type' specifies the
       target ONNX operator type, supporting 'Reduce{Max, Mean, Min, Sum}'
       right now.
    """

    inputs, attrs, outputs = op_io_info(operator)
    rank = len(block.vars[get_old_name(inputs['X'][0])].shape)
    dim = attrs['dim']
    axes = [dim if dim >= 0 else rank + dim]
    reduce_out = [outputs['Out'][0] + '@reduce_0'] if attrs[
        'reduce_all'] else outputs
    reduce_node = make_node(
        op_type,
        inputs=inputs['X'],
        outputs=reduce_out,
        axes=axes,
        keepdims=attrs['keep_dim'])
    if attrs['reduce_all'] is True:
        axes = range(rank) if attrs['keep_dim'] else range(rank - 1)
        reduce_all_node = make_node(
            op_type,
            inputs=reduce_out,
            outputs=outputs,
            axes=axes,
            keepdims=attrs['keep_dim'])
        return (reduce_node, reduce_all_node)
    return reduce_node


def reducel1_op():
    pass


def reducel2_op():
    pass


def reducelogsum_op():
    pass


def reducelogsumexp_op():
    pass


def reduceprod_op():
    pass


def reducesumsquare_op():
    pass


def reshape_op(operator, block):
    inputs, attrs, outputs = op_io_info(operator)
    shape_name = ""
    if 'Shape' in inputs and inputs['Shape'] is not None and len(inputs[
            'Shape']) > 0:
        shape_name = inputs['Shape'][0]
        # cast the shape to int64
        shape_name_cast = [shape_name + "@cast"]
        cast_node = make_node(
            'Cast', inputs=inputs['Shape'], outputs=shape_name_cast, to=7)
        reshape_node = make_node(
            'Reshape',
            inputs=[inputs['X'][0], shape_name_cast[0]],
            outputs=outputs['Out'])
        return (cast_node, reshape_node)
    elif 'ShapeTensor' in inputs and inputs['ShapeTensor'] is not None and len(
            inputs['ShapeTensor']) > 0:
        shape_name = inputs['ShapeTensor']
        return make_node(
            'Reshape',
            inputs=[inputs['X'][0], shape_name],
            outputs=outputs['Out'])
    else:
        shape_name = _tmp_name(outputs['Out'][0], "shape_var", block)
        output_shape_node = make_node(
            'Constant',
            inputs=[],
            outputs=[shape_name],
            value=make_tensor(
                name=shape_name,
                data_type=TensorProto.INT64,
                dims=[len(attrs['shape'])],
                vals=attrs['shape']))
        reshape_node = make_node(
            'Reshape',
            inputs=[inputs['X'][0], shape_name],
            outputs=outputs['Out'])
        return (output_shape_node, reshape_node)


def selu_op():
    pass


def shape_op(operator, block):
    inputs, attrs, outputs = op_io_info(operator)
    im_outputs = []
    node_list = []
    outputs_shape = [_tmp_name(outputs['Out'][0], "shape", block)]
    node_shape = make_node(
        'Shape', inputs=inputs['Input'], outputs=outputs_shape)
    node_list.append(node_shape)
    node_cast = make_node(
        'Cast',
        inputs=outputs_shape,
        outputs=outputs['Out'],
        to=PADDLE_TO_ONNX_DTYPE[core.VarDesc.VarType.FP32])
    node_list.append(node_cast)
    im_outputs.append(
        helper.make_tensor_value_info('Out', PADDLE_TO_ONNX_DTYPE[
            core.VarDesc.VarType.INT64], [2]))
    return tuple(node_list)


def size_op():
    pass


def slice_op(operator, block):
    inputs, attrs, outputs = op_io_info(operator)
    axes = attrs['axes']
    starts = attrs['starts']
    ends = attrs['ends']
    node_list = []

    name_starts = [outputs['Out'][0] + "@starts"]
    node_starts = make_node(
        'Constant',
        inputs=[],
        outputs=name_starts,
        value=make_tensor(
            name=name_starts[0],
            data_type=TensorProto.INT64,
            dims=[len(starts)],
            vals=starts))
    node_list.append(node_starts)

    name_ends = [outputs['Out'][0] + "@ends"]
    node_ends = make_node(
        'Constant',
        inputs=[],
        outputs=name_ends,
        value=make_tensor(
            name=name_ends[0],
            data_type=TensorProto.INT64,
            dims=[len(ends)],
            vals=ends))
    node_list.append(node_ends)

    name_axes = [outputs['Out'][0] + "@axes"]
    node_axes = make_node(
        'Constant',
        inputs=[],
        outputs=name_axes,
        value=make_tensor(
            name=name_axes[0],
            data_type=TensorProto.INT64,
            dims=[len(axes)],
            vals=axes))
    node_list.append(node_axes)

    node_output = make_node(
        'Slice',
        inputs=inputs['Input'] + name_starts + name_ends + name_axes,
        outputs=outputs['Out'])
    node_list.append(node_output)
    return tuple(node_list)


def nearest_interp_op(operator, block):
    inputs, attrs, outputs = op_io_info(operator)
    input_shape = block.vars[get_old_name(inputs['X'][0])].shape
    batch_size = input_shape[0]
    channels = input_shape[1]
    height = input_shape[2]
    width = input_shape[3]
    node_list = []
    im_outputs = []

    outputs_out_size_f = [outputs['Out'][0] + "@out_size_f"]
    node_out_size_f = make_node(
        'Cast', inputs=inputs['OutSize'], outputs=outputs_out_size_f, to=1)
    node_list.append(node_out_size_f)

    name_h_w = [outputs['Out'][0] + "@h_w"]
    node_h_w = make_node(
        'Constant',
        inputs=[],
        outputs=name_h_w,
        value=make_tensor(
            name=name_h_w[0],
            data_type=TensorProto.FLOAT,
            dims=[2],
            vals=[height, width]))
    node_list.append(node_h_w)

    outputs_h_w_scales = [outputs['Out'][0] + "@h_w_scales"]
    node_h_w_scales = make_node(
        'Div', inputs=outputs_out_size_f + name_h_w, outputs=outputs_h_w_scales)
    node_list.append(node_h_w_scales)

    name_b_c_scales = [outputs['Out'][0] + "@b_c_scales"]
    node_b_c_scales = make_node(
        'Constant',
        inputs=[],
        outputs=name_b_c_scales,
        value=make_tensor(
            name=name_b_c_scales[0],
            data_type=TensorProto.FLOAT,
            dims=[2],
            vals=[1, 1]))
    node_list.append(node_b_c_scales)

    outputs_scales = [outputs['Out'][0] + "@scales"]
    node_scales = make_node(
        'Concat',
        inputs=name_b_c_scales + outputs_h_w_scales,
        outputs=outputs_scales,
        axis=0)
    node_list.append(node_scales)

    outputs_resize = [outputs['Out'][0] + "@resize"]
    node_resize = make_node(
        'Resize',
        inputs=inputs['X'] + outputs_scales,
        outputs=outputs['Out'],
        mode='nearest')
    node_list.append(node_resize)
    return tuple(node_list)


def softmax_op(operator, block):
    inputs, attrs, outputs = op_io_info(operator)
    paddle_var = block.var(inputs["X"][0])
    axis = attrs['axis']
    #if axis < 0:
    #   axis = axis + len(paddle_var.shape)
    return make_node(
        'Softmax', inputs=inputs['X'], outputs=outputs['Out'], axis=axis)


def spacetodepth_op():
    pass


def split_op(operator, block):
    inputs, attrs, outputs = op_io_info(operator)
    len_sec = len(attrs['sections'])
    if len_sec > 0:
        return make_node(
            'Split',
            inputs=inputs['X'],
            outputs=outputs['Out'],
            axis=attrs['axis'],
            split=attrs['sections'])
    else:
        return make_node(
            'Split',
            inputs=inputs['X'],
            outputs=outputs['Out'],
            axis=attrs['axis'])


def squeeze_op():
    pass


def stack_op(operator, block):
    inputs, attrs, outputs = op_io_info(operator)
    axis = attrs['axis']
    concat_inputs = []
    node_list = []
    for input in inputs['X']:
        name_unsqueezed = _tmp_name(input, 'unsqueezed', block)
        concat_inputs.append(name_unsqueezed)
        node_list.append(make_node(
            'Unsqueeze',
            inputs=[input],
            outputs=[name_unsqueezed],
            axes=[axis]))
    node_list.append(make_node(
        'Concat',
        inputs=concat_inputs,
        outputs=outputs['Y'],
        axis=axis))
    return tuple(node_list)


def sub_op():
    pass


def sum_op():
    pass


def tile_op():
    pass


def topk_op():
    pass


def transpose_op(operator, block):
    inputs, attrs, outputs = op_io_info(operator)
    node = make_node(
        'Transpose',
        inputs=inputs['X'],
        outputs=outputs['Out'],
        perm=attrs['axis'])
    return node


def unsqueeze_op(operator, block):
    inputs, attrs, outputs = op_io_info(operator)
    axes = attrs['axes']
    return make_node(
        'Unsqueeze',
        inputs=inputs['X'],
        outputs=outputs['Out'],
        axes=axes)

def thresholded_relu_op(operator, block):
    inputs, attrs, outputs = op_io_info(operator)
    return make_node(
        'ThresholdedRelu',
        inputs=inputs['X'],
        outputs=outputs['Out'],
        alpha=attrs['threshold'])


def scale_op(operator, block):
    inputs, attrs, outputs = op_io_info(operator)
    scale_var_name = [outputs['Out'][0] + "@scale"]
    node_scale = onnx.helper.make_node(
        'Constant',
        inputs=[],
        outputs=scale_var_name,
        value=onnx.helper.make_tensor(
            name=scale_var_name[0] + "@const",
            data_type=onnx.TensorProto.FLOAT,
            dims=(),
            vals=[attrs['scale']]))
    bais_var_name = [outputs['Out'][0] + "@bais"]
    bais = 0.0
    if 'bais' in attrs:
        bais = float(attrs['bais'])
    node_bais = onnx.helper.make_node(
        'Constant',
        inputs=[],
        outputs=bais_var_name,
        value=onnx.helper.make_tensor(
            name=bais_var_name[0] + "@const",
            data_type=onnx.TensorProto.FLOAT,
            dims=(),
            vals=[bais]))
    paddle_var = block.var(inputs["X"][0])
    tmp_var_name = outputs['Out'][0] + "@tmp"
    shape = paddle_onnx_shape(paddle_var.shape)
    tmp_var = onnx.helper.make_tensor_value_info(
        tmp_var_name, PADDLE_TO_ONNX_DTYPE[paddle_var.dtype], shape)
    nodes = (node_scale, node_bais)
    if attrs['bias_after_scale'] == True:
        node_output_mul = onnx.helper.make_node(
            'Mul',
            inputs=[scale_var_name[0], inputs["X"][0]],
            outputs=[tmp_var_name])
        node_output_scale = onnx.helper.make_node(
            'Add',
            inputs=[bais_var_name[0], tmp_var_name],
            outputs=outputs["Out"])
        nodes += (node_output_mul, node_output_scale)
    else:
        node_output_add = onnx.helper.make_node(
            'Add',
            inputs=[bais_var_name[0], inputs["X"][0]],
            outputs=[tmp_var_name])
        node_output_mul = onnx.helper.make_node(
            'Mul',
            inputs=[scale_var_name[0], tmp_var_name],
            outputs=outputs["Out"])
        nodes += (node_output_add, node_output_mul)

    return nodes


def swish_op(operator, block):
    """
    The activation swish, x / (1 + exp(-beta * x))
    """
    inputs, attrs, outputs = op_io_info(operator)
    paddle_var = block.var(inputs["X"][0])
    shape = paddle_onnx_shape(paddle_var.shape)

    if 'beta' in attrs:
        beta = attrs['beta']
    if 'slope' in attrs:
        beta = attrs['slope']

    name_beta = [outputs['Out'][0] + "@swish_beta"]
    node_beta = onnx.helper.make_node(
        'Constant',
        inputs=[],
        outputs=name_beta,
        value=onnx.helper.make_tensor(
            name=name_beta[0] + "@const",
            data_type=onnx.TensorProto.FLOAT,
            dims=(),
            vals=[beta]))

    # var and node for beta * x
    name_beta_x = [outputs['Out'][0] + "@beta_x"]
    var_beta_x = onnx.helper.make_tensor_value_info(
        name_beta_x[0], PADDLE_TO_ONNX_DTYPE[paddle_var.dtype], shape)
    node_beta_x = onnx.helper.make_node(
        'Mul', inputs=[name_beta[0], inputs['X'][0]], outputs=name_beta_x)

    # var and node sigmoid(beta*x)
    name_sigmoid_x = [outputs['Out'][0] + "@sigmoid_x"]
    var_sigmoid_x = onnx.helper.make_tensor_value_info(
        name_sigmoid_x[0], PADDLE_TO_ONNX_DTYPE[paddle_var.dtype], shape)
    node_sigmoid_x = onnx.helper.make_node(
        'Sigmoid', inputs=name_beta_x, outputs=name_sigmoid_x)

    node_swish = onnx.helper.make_node(
        'Mul', inputs=inputs['X'] + name_sigmoid_x, outputs=outputs['Out'])
    return (node_beta, node_beta_x, node_sigmoid_x, node_swish)


def relu6_op(operator, block):
    """
    The activation function relu6, out=min(max(0,x),6)
    And you can set the threshold of activation.
    """
    inputs, attrs, outputs = op_io_info(operator)
    threshold = attrs['threshold']
    relu6_node = onnx.helper.make_node(
        'Clip',
        inputs=inputs['X'],
        outputs=outputs['Out'],
        max=threshold,
        min=0.0)
    return relu6_node


def assign_value_op(operator, block):
    inputs, attrs, outputs = op_io_info(operator)
    values = None
    data_type = None
    if 'fp32_values' in attrs and len(attrs['fp32_values']) > 0:
        values = attrs['fp32_values']
        data_type = onnx.TensorProto.FLOAT
    if 'int32_values' in attrs and len(attrs['int32_values']) > 0:
        values = attrs['int32_values']
        data_type = onnx.TensorProto.INT32

    node = onnx.helper.make_node(
        'Constant',
        inputs=[],
        outputs=outputs['Out'],
        value=onnx.helper.make_tensor(
            name=outputs['Out'][0] + "@const",
            data_type=data_type,
            dims=(),
            vals=values))
    return node


# Based on the ONNX 1.0 operator list generated on March 26th, 2018.
# Reference for paddle operator availability taken from:
#     https://github.com/PaddlePaddle/Paddle/issues/8028

node_maker = {
    # Paddle op name : (ONNX op name, modifier)
    'abs': partial(activation_ops, 'Abs'),
    # 'ArgMax', NEEDS ATTENTION.
    # 'ArgMin', NEEDS ATTENTION.
    'batch_norm': batch_norm_op,
    'cast': cast_op,
    'ceil': partial(activation_ops, 'Ceil'),
    'clip': clip_op,
    'concat': concat_op,
    'constant': constant_op,
    'conv2d': conv2d_op,
    # Need to continue the mapping below.
    'conv2d_transpose': conv2d_transpose_op,
    # 'cos': partial(activation_ops, 'Cos'),
    '': 'DepthToSpace',
    'depthwise_conv2d': conv2d_op,
    'dropout': dropout_op,
    'elementwise_add': partial(elementwise_ops, 'Add'),
    'elementwise_div': partial(elementwise_ops, 'Div'),
    'elementwise_mul': partial(elementwise_ops, 'Mul'),
    'elementwise_pow': partial(elementwise_ops, 'Pow'),
    'elementwise_sub': partial(elementwise_ops, 'Sub'),
    'elu': elu_op,
    'equal': partial(compare_ops, 'Equal'),
    'exp': partial(activation_ops, 'Exp'),
    '': 'Flatten',
    'floor': partial(activation_ops, 'Floor'),
    '': 'GRU',
    'gather': gather_op,
    'gelu': gelu_op,
    '': 'Gemm',
    '': 'GlobalLpPool',
    'greater_than': partial(compare_ops, 'Greater'),
    'hard_sigmoid': 'HardSigmoid',  # Caffe2 error
    # 'Hardmax', NEEDS ATTENTION.
    # 'InstanceNormalization', NEEDS ATTENTION.
    'layer_norm': layer_norm_op,
    'less_than': partial(compare_ops, 'Less'),
    'lrn': lrn_op,
    '': 'LSTM',
    'leaky_relu': leaky_relu_op,
    'log': partial(activation_ops, 'Log'),
    'logical_and': partial(binary_logical_ops, 'And'),
    'logical_or': partial(binary_logical_ops, 'Or'),
    'logical_not': partial(unary_logical_ops, 'Not'),
    'logical_xor': partial(binary_logical_ops, 'Xor'),
    ',': 'LogSoftmax',
    'lookup_table': lookup_table_op,
    '': 'LpNormalization',
    '': 'LpPool',
    '': 'MatMul',
    '': 'Max',
    # 'MaxPool', NEEDS ATTENTION.
    '': 'MaxRoiPool',
    'mean': ('Mean', mean_op),
    '': 'Min',
    'mul': mul_op,
    ',': 'Neg',
    'prelu': prelu_op,
    '': 'Pad',
    'pool2d': pool2d_op,
    ',': 'RNN',
    '': 'RandomNormal',
    # 'RandomNormalLike', NEEDS ATTENTION.
    # 'RandomUniform', NEEDS ATTENTION.
    # 'RandomUniformLike', NEEDS ATTENTION.
    'reciprocal': partial(activation_ops, 'Reciprocal'),
    '': 'ReduceL1',
    '': 'ReduceL2',
    ',': 'ReduceLogSum',
    ',': 'ReduceLogSumExp',
    'reduce_max': partial(reduce_ops, 'ReduceMax'),
    'reduce_mean': partial(reduce_ops, 'ReduceMean'),
    'reduce_min': partial(reduce_ops, 'ReduceMin'),
    '': partial(reduce_ops, 'ReduceProd'),  # Caffe2 error
    'reduce_sum': partial(reduce_ops, 'ReduceSum'),
    ',': 'ReduceSumSquare',
    'relu': partial(activation_ops, 'Relu'),
    '': 'Reshape',
    # 'Selu', NEEDS ATTENTION.
    '': 'Shape',
    'sigmoid': partial(activation_ops, 'Sigmoid'),
    # 'sin': partial(activation_ops, 'Sin'),
    '': 'Size',
    # 'Slice', NEEDS ATTENTION.
    'softmax': softmax_op,
    'softplus': partial(activation_ops, 'Softplus'),
    'softsign': partial(activation_ops, 'Softsign'),
    '': 'SpaceToDepth',
    '': 'Split',
    'sqrt': partial(activation_ops, 'Sqrt'),
    'stack': stack_op,
    # 'Squeeze', NEEDS ATTENTION.
    '': 'Sum',
    'tanh': partial(activation_ops, 'Tanh'),
    '': 'Tile',
    '': 'TopK',
    '': 'Transpose',
    # 'Unsqueeze', NEEDS ATTENTION.
    # 'experimental ATen'
    # ',': 'experimental Affine'
    # 'experimental ConstantFill'
    # 'experimental Crop'
    # 'experimental FC'
    # 'experimental GRUUnit'
    # 'experimental GivenTensorFill'
    # 'assign': 'experimental Identity'
    # 'experimental If'
    # ',': 'experimental ImageScaler'
    # 'experimental Loop'
    # 'experimental LoopIndexTensor'
    # 'experimental MeanVarianceNormalization'
    # 'experimental ParametricSoftplus'
    # 'experimental Scale'
    # 'experimental ScaledTanh'
    'thresholded_relu': thresholded_relu_op,
    'scale': scale_op,
    'split': split_op,
    'reshape2': reshape_op,
    'transpose2': transpose_op,
    'swish': swish_op,
    'relu6': relu6_op,
    'multiclass_nms': multiclass_nms_op,
    'prior_box': prior_box_op,
    'box_coder': box_coder_op,
    'flatten2': flatten_op,
    'assign_value': assign_value_op,
    'yolo_box': yolo_box_op,
    'slice': slice_op,
    'nearest_interp': nearest_interp_op,
    'shape': shape_op,
    'fill_constant': fill_constant_op,
    'bilinear_interp': bilinear_interp_op,
    'arg_max': arg_max_op,
    'unsqueeze2': unsqueeze_op,
    'norm': norm_op,
    'matmul': matmul_op
    # 'experimental Upsample'
}
