import paddle
from paddle.base.framework import in_dygraph_mode
from paddle.base.layer_helper import LayerHelper
from paddle import _legacy_C_ops


@paddle.jit.not_to_static
def quantize_linear(x, scale, zero_point, bit_length=8, quant_axis=-1, name=None):
    helper = LayerHelper("quantize_linear", **locals())

    attrs = ("bit_length", bit_length, "quant_axis", quant_axis)
    if in_dygraph_mode():
        return _legacy_C_ops.quantize_linear(x, scale, zero_point, *attrs)
    else:
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
    else:
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
