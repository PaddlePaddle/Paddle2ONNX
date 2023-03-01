import argparse
import os
import paddle
import numpy
import standard_model_pb2
import helper
import framework_pb2


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--paddle_model',
        required=True,
        help='Path of directory saved the paddle model, just like MobileNetV3/inference.'
    )
    parser.add_argument(
        '--save_dir',
        required=True,
        help='Path of directory to save the new exported model.')
    return parser.parse_args()


dtype_map = {
    framework_pb2.VarType.INT16: "int16",
    framework_pb2.VarType.INT32: "int32",
    framework_pb2.VarType.INT64: "int64",
    framework_pb2.VarType.FP16: "float16",
    framework_pb2.VarType.FP32: "float32",
    framework_pb2.VarType.FP64: "float64",
    framework_pb2.VarType.BOOL: "bool"
}


def convert_params(paddle_model, save_dir):
    model_file = open(paddle_model + ".pdmodel", 'rb')
    model_str = model_file.read()
    model_file.close()
    model = framework_pb2.ProgramDesc()
    prog = model.FromString(model_str)
    block_size = len(prog.blocks)

    all_val_name_set = set()
    for block in prog.blocks:
        for var in block.vars:
            if var.persistable and var.name not in ["feed", "fetch"]:
                all_val_name_set.add(var.name)
    all_val_name_set = sorted(all_val_name_set)

    params2val_dict = {}
    params_file = open(paddle_model + ".pdiparams", 'rb')
    for name in all_val_name_set:
        _ = numpy.fromfile(params_file, dtype='int32', count=1)
        _ = numpy.fromfile(params_file, dtype='int64', count=1)
        _ = numpy.fromfile(params_file, dtype='int32', count=1)
        proto_size = numpy.fromfile(params_file, dtype='int32', count=1)
        proto_str = params_file.read(proto_size[0])
        tensor_desc = framework_pb2.VarType.TensorDesc().FromString(proto_str)
        nums = 1
        for dim in tensor_desc.dims:
            nums *= dim
        raw_data = numpy.fromfile(
            params_file, dtype=dtype_map[tensor_desc.data_type], count=nums)
        params2val_dict[name] = raw_data

    layer2params_dict = {}
    for block_idex in range(block_size):
        block = prog.blocks[block_idex]
        for op_idex in range(len(block.ops)):
            op = block.ops[op_idex]
            layer_name = op.type + "_" + str(block_idex) + "_" + str(op_idex)
            for input in op.inputs:
                for argument in input.arguments:
                    if argument in params2val_dict:
                        if layer_name in layer2params_dict:
                            layer2params_dict[layer_name].append(argument)
                        else:
                            layer2params_dict[layer_name] = [argument]

    # save params to file
    params_save_path = os.path.join(save_dir, 'standard_model.params')
    file = open(params_save_path, 'wb')
    for layer_name, param_list in layer2params_dict.items():
        file.write(("layer_name:" + layer_name + '\n').encode('utf-8'))
        for param_name in param_list:
            file.write(("param_name:" + param_name + '\n').encode('utf-8'))
            np_byte = params2val_dict[param_name].tobytes()
            bytes_str = str(np_byte) + '\n'
            bytes_str_enc = bytes_str.encode()
            file.write(bytes_str_enc)
    file.close()
    print("standard params saved in: ", params_save_path)


def convert_model(paddle_model, save_dir):
    model_file = open(paddle_model + ".pdmodel", 'rb')
    model_str = model_file.read()
    model_file.close()
    model = framework_pb2.ProgramDesc()
    prog = model.FromString(model_str)
    block_size = len(prog.blocks)

    new_model = standard_model_pb2.Model()
    new_model.contributors.name.append("PaddlePaddle")
    new_model.contributors.email.append("PaddlePaddle")
    new_model.contributors.institute.append("Baidu")
    new_model.version = prog.version.version
    new_model.framework_name = "PaddlePaddle"
    new_model.framework_version = "2.4"
    new_model.model_name = "Standard_model"
    new_model.model_version = "V1.0"
    new_model.doc_url = "https://www.paddlepaddle.org.cn/"

    for block_idex in range(block_size):
        graph = new_model.graph.add()
        block = prog.blocks[block_idex]
        graph.id = block.idx
        graph.parent_idx = block.parent_idx
        for op_index in range(len(block.ops)):
            op = block.ops[op_index]
            layer_name = op.type + "_" + str(block_idex) + "_" + str(op_index)
            operator = helper.make_standard_operator(op, layer_name, block.vars)
            graph.operator_node.append(operator)

        for var in block.vars:
            var_proto = standard_model_pb2.VariableType.FromString(
                var.SerializeToString())
            graph.variable_type.append(var_proto)

        graph.forward_block_idx = block.forward_block_idx

    model_save_path = os.path.join(save_dir, "standard_model.model")
    model_str = new_model.SerializeToString()
    with open(model_save_path, "wb") as writable:
        writable.write(model_str)
    print("standard model saved in: ", model_save_path)


if __name__ == '__main__':
    args = parse_arguments()
    paddle.set_device("cpu")
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    convert_params(args.paddle_model, args.save_dir)
    convert_model(args.paddle_model, args.save_dir)
