# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
from paddle import _legacy_C_ops
from paddle.base.framework import in_dygraph_mode
from paddle.base.layer_helper import LayerHelper


@paddle.jit.not_to_static
def quantize_linear(x, scale, zero_point, bit_length=8, quant_axis=-1, name=None):
    helper = LayerHelper("quantize_linear", **locals())

    attrs = ("bit_length", bit_length, "quant_axis", quant_axis)
    if in_dygraph_mode():
        return _legacy_C_ops.quantize_linear(x, scale, zero_point, *attrs)
    output = helper.create_variable_for_type_inference(dtype=x.dtype)

    inputs = {"X": x, "Scale": scale, "ZeroPoint": zero_point}
    outputs = {"Y": output}

    helper.append_op(
        type="quantize_linear",
        inputs=inputs,
        attrs={"bit_length": bit_length, "quant_axis": quant_axis},
        outputs=outputs,
    )
    output.stop_gradient = True
    return output


@paddle.jit.not_to_static
def dequantize_linear(x, scale, zero_point, bit_length=8, quant_axis=-1, name=None):
    helper = LayerHelper("dequantize_linear", **locals())

    attrs = ("bit_length", bit_length, "quant_axis", quant_axis)
    if in_dygraph_mode():
        return _legacy_C_ops.dequantize_linear(x, scale, zero_point, *attrs)
    output = helper.create_variable_for_type_inference(dtype=x.dtype)

    inputs = {"X": x, "Scale": scale, "ZeroPoint": zero_point}
    outputs = {"Y": output}

    helper.append_op(
        type="dequantize_linear",
        inputs=inputs,
        attrs={"bit_length": bit_length, "quant_axis": quant_axis},
        outputs=outputs,
    )
    output.stop_gradient = True
    return output
