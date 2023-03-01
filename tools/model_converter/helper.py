from typing import ItemsView
from standard_model_pb2 import OperatorNode, AttrType
import framework_pb2

paddle_int_2_str_map = {
    0: "BOOL",
    1: "INT16",
    2: "INT32",
    3: "INT64",
    4: "FLOAT16",
    5: "FLOAT32",
    6: "FLOAT64",
    19: "SIZE_T",
    20: "UINT8",
    21: "INT8",
    22: "BF16",
    23: "COMPLEX64",
    24: "COMPLEX128",
}

standard_str_2_int_map = {
    "UNDEFINED": 1,
    "BINARY": 2,
    "INT2": 3,
    "INT4": 4,
    "INT8": 5,
    "INT16": 6,
    "INT32": 7,
    "INT64": 8,
    "UINT2": 9,
    "UINT4": 10,
    "UINT8": 11,
    "UINT16": 12,
    "UINT32": 13,
    "UINT64": 14,
    "FLOAT16": 15,
    "FLOAT32": 16,
    "FLOAT64": 17,
    "BOOL": 18,
    "STRING": 19,
    "COMPLEX64": 20,
    "COMPLEX128": 21
}


def make_standard_operator(op, name, all_vars):
    operator = OperatorNode()
    operator.operator_type = op.type
    operator.name = name
    operator.doc_string = op.type + " OP."
    operator.definition = op.type + " OP."
    for input in op.inputs:
        for argument in input.arguments:
            variable_type = operator.inputs[input.parameter].variable_type.add()
            variable_type.name = argument
            variable_type.data_type = 1
            variable_type.is_persitable = False
            for var in all_vars:
                if var.name == argument:
                    variable_type.is_persitable = var.persistable
                    if var.type.type == framework_pb2.VarType.LOD_TENSOR:
                        variable_type.data_type = standard_str_2_int_map[
                            paddle_int_2_str_map[
                                var.type.lod_tensor.tensor.data_type]]
                    else:
                        variable_type.data_type = var.type.type
                    break

    for output in op.outputs:
        for argument in output.arguments:
            variable_type = operator.outputs[
                output.parameter].variable_type.add()
            variable_type.name = argument
            variable_type.data_type = 1
            variable_type.is_persitable = False
            for var in all_vars:
                if var.name == argument:
                    variable_type.is_persitable = var.persistable
                    if var.type.type == framework_pb2.VarType.LOD_TENSOR:
                        variable_type.data_type = standard_str_2_int_map[
                            paddle_int_2_str_map[
                                var.type.lod_tensor.tensor.data_type]]
                    else:
                        variable_type.data_type = var.type.type
                    break

    for paddle_attr in op.attrs:
        if paddle_attr.name in ["op_callstack"]:
            continue
        attr_get = OperatorNode.Attr.FromString(paddle_attr.SerializeToString())
        variable_type = operator.attribute[paddle_attr.name].CopyFrom(attr_get)
    return operator


def make_paddle_operator(op):
    operator = framework_pb2.OpDesc()
    operator.type = op.operator_type
    for key, values in op.inputs.items():
        input_var = operator.inputs.add()
        input_var.parameter = key
        for value in values.variable_type:
            input_var.arguments.append(value.name)
    for key, values in op.outputs.items():
        output_var = operator.outputs.add()
        output_var.parameter = key
        for value in values.variable_type:
            output_var.arguments.append(value.name)

    for key, value in op.attribute.items():
        attr_get = framework_pb2.OpDesc.Attr.FromString(value.SerializeToString(
        ))
        operator.attrs.append(attr_get)
    return operator
