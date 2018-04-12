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

from onnx.helper import make_node
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
- lookup_table
- tanh
- lstm
- sequence_pool
- lookup_table
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


def abs_op():
    pass


def add_op(inputs, attrs, outputs):
    return make_node(
        'Add',
        inputs=inputs['X'] + inputs['Y'],
        outputs=outputs['Out'],
        axis=attrs['axis'],
        broadcast=1)


def and_op():
    """
    Need to support broadcast.
    """
    pass


def argmax_op():
    pass


def argmin_op():
    pass


def averagepool_op():
    """
    Need to support more pad mode.
    """
    pass


def batchnorm_op(inputs, attrs, outputs):
    bn_op = make_node(
        'BatchNormalization',
        inputs=inputs['X'] + inputs['Scale'] + inputs['Bias'] + inputs['Mean'] +
        inputs['Variance'],
        outputs=outputs['Y'],
        is_test=attrs['is_test'],
        epsilon=attrs['epsilon'],
        momentum=attrs['momentum'])
    return bn_op


def cast_op():
    pass


def ceil_op():
    pass


def clip_op():
    pass


def concat_op():
    pass


def constant_op():
    pass


def conv_op(inputs, attrs, outputs):
    conv2d = make_node(
        'Conv',
        inputs=inputs['Input'] + inputs['Filter'],
        outputs=outputs['Output'],
        dilations=attrs['dilations'],
        kernel_shape=attrs['kernel_shape'][2:],
        strides=attrs['strides'],
        group=attrs['groups'],
        pads=attrs['paddings'])
    return conv2d


def convtranspose_op():
    pass


def depthtospace_op():
    pass


def div_op():
    pass


def dropout_op():
    pass


def elu_op():
    pass


def equal_op():
    pass


def dropout_op():
    pass


def exp_op():
    pass


def flatten_op():
    pass


def floor_op():
    pass


def gru_op():
    pass


def gather_op():
    pass


def gemm_op():
    pass


def globalaveragepool_op():
    pass


def globallppool_op():
    pass


def globalmaxpool_op():
    pass


def greater_op():
    pass


def hardsigmoid_op():
    pass


def hardmax_op():
    pass


def instancenormalization_op():
    pass


def lrn_op():
    pass


def lstm_op():
    pass


def leakyrelu_op():
    pass


def less_op():
    pass


def log_op():
    pass


def logsoftmax_op():
    pass


def lpnormalization_op():
    pass


def lppool_op():
    pass


def matmul_op(inputs, attrs, outputs):
    return make_node(
        'MatMul', inputs=inputs['X'] + inputs['Y'], outputs=outputs['Out'])


def max_op():
    pass


def maxpool_op():
    """
    Need to support broadcast.
    """
    pass


def maxroipool_op():
    pass


def mean_op():
    pass


def min_op():
    pass


def mul_op():
    pass


def neg_op():
    pass


def not_op():
    """
    Need to support broadcast.
    """
    pass


def or_op():
    """
    Need to support broadcast.
    """
    pass


def prelu_op():
    pass


def pad_op():
    pass


def pool2d_op(inputs, attrs, outputs):
    if attrs['global_pooling'] is False:
        op_type = {'max': 'MaxPool', 'ave': 'AveragePool'}
        pool2d = make_node(
            op_type[attrs['pooling_type']],
            inputs=inputs['X'],
            outputs=outputs['Out'],
            kernel_shape=attrs['ksize'],
            strides=attrs['strides'],
            pads=attrs['paddings'] + attrs['paddings'], )
    else:
        op_type = {'max': 'GlobalMaxPool', 'ave': 'GlobalAveragePool'}
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


def reciprocal_op():
    pass


def reducel1_op():
    pass


def reducel2_op():
    pass


def reducelogsum_op():
    pass


def reducelogsumexp_op():
    pass


def reducemax_op():
    pass


def reducemean_op():
    pass


def reducemin_op():
    pass


def reduceprod_op():
    pass


def reducesum_op():
    pass


def reducesumsquare_op():
    pass


def relu_op(inputs, attrs, outputs):
    return make_node('Relu', inputs=inputs['X'], outputs=outputs['Out'])


def reshape_op():
    pass


def selu_op():
    pass


def shape_op():
    pass


def sigmoid_op():
    pass


def size_op():
    pass


def slice_op():
    pass


def softmax_op(inputs, attrs, outputs):
    return make_node('Softmax', inputs=inputs['X'], outputs=outputs['Out'])


def softplus_op():
    pass


def softsign_op():
    pass


def spacetodepth_op():
    pass


def split_op():
    pass


def sqrt_op():
    pass


def squeeze_op():
    pass


def sub_op():
    pass


def sum_op():
    pass


def tanh_op():
    pass


def tile_op():
    pass


def topk_op():
    pass


def transpose_op():
    pass


def unsqueeze_op():
    pass


def xor_op():
    """
    Need to support broadcast.
    """
    pass


# Based on the ONNX 1.0 operator list generated on March 26th, 2018.
# Reference for paddle operator availability taken from:
#     https://github.com/PaddlePaddle/Paddle/issues/8028

node_maker = {
    # Paddle op name : (ONNX op name, modifier)
    'abs': ('Abs', abs_op),
    'elementwise_add': add_op,

    # '': 'And', # ?
    # 'ArgMax', NEEDS ATTENTION.
    # 'ArgMin', NEEDS ATTENTION.
    '': ('AveragePool', averagepool_op),
    'batch_norm': batchnorm_op,
    'cast': ('Cast', cast_op),
    # 'Ceil', NEEDS ATTENTION.
    'cast': ('Clip', clip_op),
    'concat': ('Concat', concat_op),
    ',': ('Constant', constant_op),
    'conv2d': conv_op,

    # Need to continue the mapping below.
    '': 'ConvTranspose',
    '': 'DepthToSpace',
    '': 'Div',
    '': 'Dropout',
    '': 'Elu',
    '': 'Equal',
    '': 'Exp',
    '': 'Flatten',
    # 'Floor', NEEDS ATTENTION.
    '': 'GRU',
    '': 'Gather',
    '': 'Gemm',
    '': 'GlobalAveragePool',
    '': 'GlobalLpPool',
    '': 'GlobalMaxPool',
    '': 'Greater',
    '': 'HardSigmoid',
    # 'Hardmax', NEEDS ATTENTION.
    # 'InstanceNormalization', NEEDS ATTENTION.
    '': 'LRN',
    '': 'LSTM',
    '': 'LeakyRelu',
    '': 'Less',
    '': 'Log',
    ',': 'LogSoftmax',
    '': 'LpNormalization',
    '': 'LpPool',
    '': 'MatMul',
    '': 'Max',
    # 'MaxPool', NEEDS ATTENTION.
    '': 'MaxRoiPool',
    'mean': ('Mean', mean_op),
    '': 'Min',
    'mul': matmul_op,
    ',': 'Neg',
    '': 'Not',
    '': 'Or',
    '': 'PRelu',
    '': 'Pad',
    'pool2d': pool2d_op,
    '': 'Pow',
    ',': 'RNN',
    '': 'RandomNormal',
    # 'RandomNormalLike', NEEDS ATTENTION.
    # 'RandomUniform', NEEDS ATTENTION.
    # 'RandomUniformLike', NEEDS ATTENTION.
    '': 'Reciprocal',
    '': 'ReduceL1',
    '': 'ReduceL2',
    ',': 'ReduceLogSum',
    ',': 'ReduceLogSumExp',
    '': 'ReduceMax',
    '': 'ReduceMean',
    '': 'ReduceMin',
    # 'ReduceProd', NEEDS ATTENTION.
    '': 'ReduceSum',
    ',': 'ReduceSumSquare',
    'relu': relu_op,
    '': 'Reshape',
    # 'Selu', NEEDS ATTENTION.
    '': 'Shape',
    '': 'Sigmoid',
    '': 'Size',
    # 'Slice', NEEDS ATTENTION.
    'softmax': softmax_op,
    '': 'Softplus',
    '': 'Softsign',
    '': 'SpaceToDepth',
    '': 'Split',
    '': 'Sqrt',
    # 'Squeeze', NEEDS ATTENTION.
    'elementwise_sub': ('Sub', sub_op),
    '': 'Sum',
    '': 'Tanh',
    '': 'Tile',
    '': 'TopK',
    '': 'Transpose',
    # 'Unsqueeze', NEEDS ATTENTION.
    '': 'Xor',
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
    # 'experimental ThresholdedRelu'
    # 'experimental Upsample'
}
