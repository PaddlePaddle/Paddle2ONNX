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

import numpy as np
from onnx import helper, onnx_pb, TensorProto
import paddle.fluid.core as core
from paddle.fluid.executor import _fetch_var as fetch_var
import sys


def paddle_variable_to_onnx_tensor(paddle_var_name, block):
    # TODO(varunarora): Need to do this only in the case of VarType.LOD_TENSOR.
    paddle_var = block.var(paddle_var_name)
    shape = paddle_onnx_shape(paddle_var.shape)
    return helper.make_tensor_value_info(
        paddle_var_name, PADDLE_TO_ONNX_DTYPE[paddle_var.dtype], shape)


def paddle_onnx_shape(paddle_shape):
    """ Convert shape info from paddle to onnx
    """

    onnx_shape = np.array(list(paddle_shape))
    onnx_shape[onnx_shape < 0] = 1
    output_shape = tuple(onnx_shape)
    # python3 do not have the int64
    python_version = sys.version
    if int(python_version[0]) == 3:
        output_shape = [int(sp) for sp in output_shape]
    return output_shape


def paddle_onnx_weight(var, scope):
    data = fetch_var(str(var.name), scope)
    weight = helper.make_tensor(
        name=var.name,
        dims=var.shape,
        data_type=PADDLE_TO_ONNX_DTYPE[var.dtype],
        vals=data.flatten().tolist())
    value_info = helper.make_tensor_value_info(
        var.name, PADDLE_TO_ONNX_DTYPE[var.dtype], var.shape)
    return weight, value_info


PADDLE_TO_ONNX_DTYPE = {
    core.VarDesc.VarType.FP32: onnx_pb.TensorProto.FLOAT,
    core.VarDesc.VarType.FP64: onnx_pb.TensorProto.DOUBLE,
    # '': onnx_pb.TensorProto.DOUBLE,
    core.VarDesc.VarType.INT32: onnx_pb.TensorProto.INT32,
    core.VarDesc.VarType.INT16: onnx_pb.TensorProto.INT16,
    # '': onnx_pb.TensorProto.INT8,
    # '': onnx_pb.TensorProto.UINT8,
    core.VarDesc.VarType.INT16: onnx_pb.TensorProto.UINT16,
    core.VarDesc.VarType.INT64: onnx_pb.TensorProto.INT64,
    # '': onnx_pb.TensorProto.STRING,
    # '': onnx_pb.TensorProto.COMPLEX64,
    # '': onnx_pb.TensorProto.COMPLEX128,
    core.VarDesc.VarType.BOOL: onnx_pb.TensorProto.BOOL
}

PADDLE_DTYPE_DICT = {
    'float32': core.VarDesc.VarType.FP32,
    'float64': core.VarDesc.VarType.FP64,
    'int32': core.VarDesc.VarType.INT32,
    'int64': core.VarDesc.VarType.INT64,
    'bool': core.VarDesc.VarType.BOOL
}
