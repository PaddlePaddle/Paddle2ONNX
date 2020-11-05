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

import numpy as np
import paddle.fluid.core as core
from paddle2onnx.onnx_helper import helper
from paddle2onnx.onnx_helper.onnx_pb import TensorProto

ONNX = TensorProto

DTYPE_PADDLE_ONNX_MAP = {
    TensorProto.FLOAT: core.VarDesc.VarType.FP32,
    TensorProto.DOUBLE: core.VarDesc.VarType.FP64,
    TensorProto.INT16: core.VarDesc.VarType.INT16,
    TensorProto.INT32: core.VarDesc.VarType.INT32,
    TensorProto.INT64: core.VarDesc.VarType.INT64,
    TensorProto.BOOL: core.VarDesc.VarType.BOOL,
    core.VarDesc.VarType.FP32: TensorProto.FLOAT,
    core.VarDesc.VarType.FP64: TensorProto.DOUBLE,
    core.VarDesc.VarType.INT16: TensorProto.INT16,
    core.VarDesc.VarType.INT32: TensorProto.INT32,
    core.VarDesc.VarType.INT64: TensorProto.INT64,
    core.VarDesc.VarType.BOOL: TensorProto.BOOL,
}

DTYPE_PADDLE_NUMPY_MAP = {
    np.float32: core.VarDesc.VarType.FP32,
    np.float64: core.VarDesc.VarType.FP64,
    np.int16: core.VarDesc.VarType.INT16,
    np.int32: core.VarDesc.VarType.INT32,
    np.int64: core.VarDesc.VarType.INT64,
    np.bool: core.VarDesc.VarType.BOOL,
    core.VarDesc.VarType.FP32: np.float,
    core.VarDesc.VarType.FP64: np.float64,
    core.VarDesc.VarType.INT16: np.int16,
    core.VarDesc.VarType.INT32: np.int32,
    core.VarDesc.VarType.INT64: np.int64,
    core.VarDesc.VarType.BOOL: np.bool
}
