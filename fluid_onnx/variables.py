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
from onnx import helper, onnx_pb2, TensorProto
import paddle.fluid.core as core
from paddle.fluid.executor import fetch_var


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
    onnx_shape[onnx_shape < 0] = 0
    return tuple(onnx_shape)


def paddle_onnx_weight(var, scope):
    data = fetch_var(var.name, scope)
    weight = helper.make_tensor(
        name=var.name,
        dims=var.shape,
        data_type=PADDLE_TO_ONNX_DTYPE[var.dtype],
        vals=data.flatten().tolist())
    value_info = helper.make_tensor_value_info(
        var.name, PADDLE_TO_ONNX_DTYPE[var.dtype], var.shape)
    return weight, value_info


PADDLE_TO_ONNX_DTYPE = {
    core.VarDesc.VarType.FP32: onnx_pb2.TensorProto.FLOAT,
    core.VarDesc.VarType.FP64: onnx_pb2.TensorProto.DOUBLE,
    # '': onnx_pb2.TensorProto.DOUBLE,
    core.VarDesc.VarType.INT32: onnx_pb2.TensorProto.INT32,
    core.VarDesc.VarType.INT16: onnx_pb2.TensorProto.INT16,
    # '': onnx_pb2.TensorProto.INT8,
    # '': onnx_pb2.TensorProto.UINT8,
    core.VarDesc.VarType.INT16: onnx_pb2.TensorProto.UINT16,
    core.VarDesc.VarType.INT64: onnx_pb2.TensorProto.INT64,
    # '': onnx_pb2.TensorProto.STRING,
    # '': onnx_pb2.TensorProto.COMPLEX64,
    # '': onnx_pb2.TensorProto.COMPLEX128,
    core.VarDesc.VarType.BOOL: onnx_pb2.TensorProto.BOOL
}
