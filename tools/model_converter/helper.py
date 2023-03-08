from typing import ItemsView
from typing_extensions import Self
from standard_model_pb2 import OperatorNode, AttrType
import framework_pb2

dtype_map = {
    framework_pb2.VarType.INT16: "int16",
    framework_pb2.VarType.INT32: "int32",
    framework_pb2.VarType.INT64: "int64",
    framework_pb2.VarType.FP16: "float16",
    framework_pb2.VarType.FP32: "float32",
    framework_pb2.VarType.FP64: "float64",
    framework_pb2.VarType.BOOL: "bool"
}

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


def make_standard_operator(op, name, all_vars, params2val_dict):
    operator = OperatorNode()
    operator.operator_type = op.type
    operator.name = name
    operator.doc_string = op.type + " OP."
    operator.definition = op.type + " OP."
    for input in op.inputs:
        operator.input[input.parameter]
        for argument in input.arguments:
            variable_type = operator.input[input.parameter].variable_type.add()
            variable_type.name = argument
            variable_type.data_type = 1
            variable_type.is_persitable = False
            variable_type.tensor.name = argument
            variable_type.tensor.format = "NCHW"
            variable_type.tensor.data_type = 16
            variable_type.tensor.shape.unknown = True
            for var in all_vars:
                if var.name == argument:
                    variable_type.is_persitable = var.persistable
                    if var.type.type == framework_pb2.VarType.LOD_TENSOR:
                        variable_type.data_type = standard_str_2_int_map[
                            paddle_int_2_str_map[
                                var.type.lod_tensor.tensor.data_type]]
                        variable_type.tensor.data_type = variable_type.data_type
                        variable_type.tensor.shape.unknown = False
                        shape_dims = variable_type.tensor.shape.dim
                        for dim_index in range(
                                len(var.type.lod_tensor.tensor.dims)):
                            dim = var.type.lod_tensor.tensor.dims[dim_index]
                            shape_dim = shape_dims.add()
                            shape_dim.name = "dim_" + str(dim_index)
                            shape_dim.size = dim
                        if variable_type.is_persitable:
                            content = variable_type.tensor.content
                            numpy_array = params2val_dict[argument].reshape(
                                -1).tolist()
                            if len(numpy_array) >= 2:
                                content.add().f = numpy_array[0]
                                content.add().f = numpy_array[1]
                    else:
                        assert (False, "Unsupported var type")
                    break

    for output in op.outputs:
        operator.output[output.parameter]
        for argument in output.arguments:
            variable_type = operator.output[output.parameter].variable_type.add(
            )
            variable_type.name = argument
            variable_type.data_type = 1
            variable_type.is_persitable = False
            variable_type.tensor.name = argument
            variable_type.tensor.format = "NCHW"
            variable_type.tensor.data_type = 16
            variable_type.tensor.shape.unknown = True
            for var in all_vars:
                if var.name == argument:
                    variable_type.is_persitable = var.persistable
                    if var.type.type == framework_pb2.VarType.LOD_TENSOR:
                        variable_type.data_type = standard_str_2_int_map[
                            paddle_int_2_str_map[
                                var.type.lod_tensor.tensor.data_type]]
                        variable_type.tensor.data_type = variable_type.data_type
                        variable_type.tensor.shape.unknown = False
                        shape_dims = variable_type.tensor.shape.dim
                        for dim_index in range(
                                len(var.type.lod_tensor.tensor.dims)):
                            dim = var.type.lod_tensor.tensor.dims[dim_index]
                            shape_dim = shape_dims.add()
                            shape_dim.name = "dim_" + str(dim_index)
                            shape_dim.size = dim
                    else:
                        assert (False, "Unsupported var type")
                    break

    for paddle_attr in op.attrs:
        if paddle_attr.name in ["op_callstack"]:
            continue
        operator.attribute[paddle_attr.name].name = paddle_attr.name
        operator.attribute[paddle_attr.name].type = paddle_attr.type
        if paddle_attr.type == 0:
            operator.attribute[paddle_attr.name].val.i = paddle_attr.i
        elif paddle_attr.type == 1:
            operator.attribute[paddle_attr.name].val.f = paddle_attr.f
        elif paddle_attr.type == 2:
            operator.attribute[paddle_attr.name].val.s = paddle_attr.s
        elif paddle_attr.type == 3:
            for val in paddle_attr.ints:
                operator.attribute[paddle_attr.name].list.add().i = val
        elif paddle_attr.type == 4:
            for val in paddle_attr.floats:
                operator.attribute[paddle_attr.name].list.add().f = val
        elif paddle_attr.type == 5:
            for val in paddle_attr.strings:
                operator.attribute[paddle_attr.name].list.add().s = val
        elif paddle_attr.type == 6:
            operator.attribute[paddle_attr.name].val.b = paddle_attr.b
        elif paddle_attr.type == 7:
            for val in paddle_attr.bools:
                operator.attribute[paddle_attr.name].list.add().b = val
        elif paddle_attr.type == 8:
            operator.attribute[
                paddle_attr.name].val.block_idx = paddle_attr.block_idx
        elif paddle_attr.type == 9:
            operator.attribute[paddle_attr.name].val.l = paddle_attr.l
        elif paddle_attr.type == 11:
            for val in paddle_attr.longs:
                operator.attribute[paddle_attr.name].list.add().l = val
        elif paddle_attr.type == 12:
            for val in paddle_attr.float64s:
                operator.attribute[paddle_attr.name].list.add().float64 = val
        elif paddle_attr.type == 15:
            operator.attribute[
                paddle_attr.name].val.float64 = paddle_attr.float64
        else:
            print("unsupported paddle attr: ", paddle_attr)
    return operator


def make_paddle_operator(op):
    operator = framework_pb2.OpDesc()
    operator.type = op.operator_type
    for key, values in op.input.items():
        input_var = operator.inputs.add()
        input_var.parameter = key
        for value in values.variable_type:
            input_var.arguments.append(value.name)
    for key, values in op.output.items():
        output_var = operator.outputs.add()
        output_var.parameter = key
        for value in values.variable_type:
            output_var.arguments.append(value.name)

    for name, attr in op.attribute.items():
        paddle_attr = operator.attrs.add()
        paddle_attr.name = name
        paddle_attr.type = attr.type
        if attr.type == 0:
            paddle_attr.i = attr.val.i
        elif attr.type == 1:
            paddle_attr.f = attr.val.f
        elif attr.type == 2:
            paddle_attr.s = attr.val.s
        elif attr.type == 3:
            for val in attr.list:
                paddle_attr.ints.append(val.i)
        elif attr.type == 4:
            for val in attr.list:
                paddle_attr.floats.append(val.f)
        elif attr.type == 5:
            for val in attr.list:
                paddle_attr.strings.append(val.s)
        elif attr.type == 6:
            paddle_attr.b = attr.val.b
        elif attr.type == 7:
            for val in attr.list:
                paddle_attr.bools.append(val.b)
        elif attr.type == 8:
            paddle_attr.block_idx = attr.val.block_idx
        elif attr.type == 9:
            paddle_attr.l = attr.val.l
        elif attr.type == 11:
            for val in attr.list:
                paddle_attr.longs.append(val.l)
        elif attr.type == 12:
            for val in attr.list:
                paddle_attr.float64s.append(val.float64)
        elif attr.type == 15:
            paddle_attr.float64 = attr.val.float64
        else:
            print("unsupported standard attr: ", attr)

    return operator
