import argparse
from ast import Assert
import os
from xmlrpc.server import SimpleXMLRPCDispatcher
import paddle
import numpy
import standard_model_pb2
import helper
import framework_pb2
import six


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--standard_model',
        required=True,
        help='Path of directory saved the paddle model, just like standard_model/model.'
    )
    parser.add_argument(
        '--save_dir',
        required=False,
        default=None,
        help='Path of directory to save the new exported model.')
    return parser.parse_args()


class StandardModel(object):
    def __init__(self, model_path):
        self.model_path = model_path
        self.params2val_dict = dict()
        self.layer2params_dict = dict()
        self.all_var_names = set()
        self.prog = None
        self.load_params()
        self.load_model()

    def load_params(self):
        params_file_path = self.model_path + ".params"
        file = open(params_file_path, 'rb')
        layer_name = None
        numpy_val = None
        for line in file.readlines():
            is_val = False
            if line.count(b'_name:'):
                line = line.decode()
            else:
                is_val = True

            if is_val:
                bytes_np_dec = line.decode('unicode-escape').encode(
                    'ISO-8859-1')[2:-2]
                numpy_val = numpy.frombuffer(bytes_np_dec, dtype="float32")
            if not is_val:
                line = line.strip().split(':')
                if line[0] == "layer_name":
                    layer_name = line[1]
                    self.layer2params_dict[layer_name] = []
                if line[0] == "param_name":
                    param_name = line[1]
                    self.layer2params_dict[layer_name].append(param_name)
                    self.all_var_names.add(param_name)
            else:
                self.params2val_dict[param_name] = numpy_val
        file.close()
        self.all_var_names = sorted(self.all_var_names)

    def load_model(self):
        standard_model_file_path = self.model_path + ".model"
        standard_model_model_file = open(standard_model_file_path, 'rb')
        standard_model_str = standard_model_model_file.read()
        standard_model_model_file.close()
        self.prog = standard_model_pb2.Model().FromString(standard_model_str)
        for graph in self.prog.graph:
            for var in graph.variable_type:
                if var.name in self.params2val_dict:
                    shape = []
                    for dim in var.tensor.shape.dim:
                        shape.append(dim.size)
                    self.params2val_dict[var.name] = self.params2val_dict[
                        var.name].reshape(shape)
                    key = self.get_dict_key(helper.standard_str_2_int_map,
                                            var.data_type)
                    self.params2val_dict[var.name] = self.params2val_dict[
                        var.name].astype(key.lower())

    def save_paddle_params(self, save_dir):
        all_var_names = set()
        for _, params in self.layer2params_dict.items():
            for param in params:
                all_var_names.add(param)
        all_var_names = sorted(all_var_names)

        model_file_path = os.path.join(save_dir, 'model.pdmodel')
        with open(model_file_path, 'rb') as model_file:
            model_str = model_file.read()
            paddle_model = framework_pb2.ProgramDesc().FromString(model_str)

        params_save_path = os.path.join(save_dir, 'model.pdiparams')
        fp = open(params_save_path, 'wb')
        for name in all_var_names:
            val = self.params2val_dict[name]
            type_str = None
            for block in paddle_model.blocks:
                for b_var in block.vars:
                    if b_var.name == name:
                        dims = b_var.type.lod_tensor.tensor.dims
                        type_str = helper.paddle_int_2_str_map[
                            b_var.type.lod_tensor.tensor.data_type]
                        val = val.reshape(dims)
                        break
            shape = val.shape
            if len(shape) == 0:
                shape = [1]
            numpy.array([0], dtype='int32').tofile(fp)
            numpy.array([0], dtype='int64').tofile(fp)
            numpy.array([0], dtype='int32').tofile(fp)
            tensor_desc = framework_pb2.VarType.TensorDesc()
            key = self.get_dict_key(helper.dtype_map, type_str.lower())
            tensor_desc.data_type = key
            tensor_desc.dims.extend(shape)
            desc_size = tensor_desc.ByteSize()
            numpy.array([desc_size], dtype='int32').tofile(fp)
            fp.write(tensor_desc.SerializeToString())
            val.tofile(fp)
        fp.close()
        print("paddle params saved in: ", params_save_path)

    def get_dict_key(self, dic, value):
        keys = list(dic.keys())
        values = list(dic.values())
        idx = values.index(value)
        key = keys[idx]
        return key

    def convert_model(self, save_dir):
        paddle_model = framework_pb2.ProgramDesc()
        for graph in self.prog.graph:
            block = paddle_model.blocks.add()
            block.idx = graph.id
            block.parent_idx = graph.parent_idx
            block.forward_block_idx = graph.forward_block_idx
            for op in graph.operator_node:
                operator = helper.make_paddle_operator(op)
                block.ops.append(operator)

            for variable_type in graph.variable_type:
                var = block.vars.add()
                var.name = variable_type.name
                var.type.type = variable_type.type
                var.persistable = variable_type.is_persitable
                key = self.get_dict_key(helper.standard_str_2_int_map,
                                        variable_type.data_type)
                var.type.lod_tensor.tensor.data_type = self.get_dict_key(
                    helper.paddle_int_2_str_map, key)
                if var.type.type == framework_pb2.VarType.LOD_TENSOR_ARRAY:
                    var.type.tensor_array.tensor.data_type = self.get_dict_key(
                        helper.paddle_int_2_str_map, key)
                for dim in variable_type.tensor.shape.dim:
                    var.type.lod_tensor.tensor.dims.append(dim.size)

        paddle_model_file_path = os.path.join(save_dir, 'model.pdmodel')
        paddle_model_str = paddle_model.SerializeToString()
        with open(paddle_model_file_path, "wb") as writable:
            writable.write(paddle_model_str)
        print("paddle model saved in: ", paddle_model_file_path)

    def convert_to_paddle_model(self, save_dir):
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        self.convert_model(save_dir)
        self.save_paddle_params(save_dir)

    def model(self):
        return self.prog

    def graph(self):
        return self.prog.graph

    def operator_node(self, node_index=0, block_index=0):
        Assert(block_index >= 0 and block_index < len(self.prog.graph),
               "block_idex must be in range [0, " + str(len(self.prog.graph)) +
               "]")
        Assert(node_index >= 0 and
               node_index < len(self.prog.graph[block_index].operator_node),
               "node_index must be in range [0, " +
               str(len(self.prog.graph[block_index].operator_node)) + "]")
        return self.prog.graph[block_index].operator_node[node_index]

    def variable_type(self, var):
        if isinstance(var, six.string_types):
            find_var = None
            for vars in self.prog.graph[0].variable_type:
                if vars.name == var:
                    find_var = vars
            if find_var is None:
                print("Input var is not found: ", var)
            else:
                return find_var

        elif isinstance(var, int):
            Assert(var >= 0 and var < len(self.prog.graph[0].variable_type),
                   "tensor must be in range [0, " +
                   str(len(self.prog.graph[0].variable_type)) + "]")
            return self.prog.graph[0].variable_type[var]
        else:
            Assert(False, "Please inter a weight name or weight index")

    def print_all_tensors(self):
        for layer_name, param_names in self.layer2params_dict.items():
            print(layer_name)
            for param in param_names:
                print("   ", param, self.params2val_dict[param].shape)

    def tensor_str(self, tensor_index):
        tensor_str = str(
            model.operator_node(tensor_index).input["Bias"].variable_type[0]
            .tensor)
        tensor_str += "int32_data:\n"
        tensor_str += "uint32_data:\n"
        tensor_str += "int64_data:\n"
        tensor_str += "uint64_data:\n"
        tensor_str += "float_data:\n"
        tensor_str += "double_data:\n"
        tensor_str += "bool_data:\n"
        tensor_str += "string_data:\n"
        return tensor_str

    def tensor_val(self, tensor):
        if isinstance(tensor, six.string_types):
            if tensor in self.params2val_dict:
                return self.params2val_dict[tensor]
            else:
                Assert(False, "Input tensor is not found: " + tensor)
        elif isinstance(tensor, int):
            Assert(tensor >= 0 and tensor < len(self.params2val_dict.keys()),
                   "tensor must be in range [0, " +
                   str(len(self.params2val_dict.keys())) + "]")
            tensor_name = self.all_var_names[tensor]
            return self.params2val_dict[tensor_name]
        else:
            Assert(False, "Please inter a weight name or weight index")


if __name__ == '__main__':
    args = parse_arguments()
    paddle.set_device("cpu")
    model = StandardModel(args.standard_model)
    # print("*" * 20)
    # print("print model: ")
    # print(model.model())
    # print("*" * 20)
    # print("print graph: ")
    # print(model.graph())
    # print("*" * 20)
    # print("print contributors: ")
    # print(model.model().contributors)
    # print("*" * 20)
    # print("print node_index 21: ")
    # print(model.operator_node(21))
    # print("*" * 20)
    # print("print node_index 21 input variable_type: ")
    # print(model.operator_node(21).input["Bias"].variable_type)
    # print("*" * 20)
    # print("print node_index 21 attribute: ")
    # print(model.operator_node(21).attribute["data_layout"])
    # print("*" * 20)
    # print("print node_index 21 data_type: ")
    # print("data_type: ",model.operator_node(21).input["Bias"].variable_type[0].data_type)
    # print("*" * 20)
    # print("print the tensor of node_index 21: ")
    # print(model.tensor_str(21))
    # print("*" * 20)
    # print("print node_index 21 tensor shape: ")
    # print(model.operator_node(21).input["Bias"].variable_type[0].tensor.shape)
    # print("*" * 20)
    # print("print node_index 21 tensor shape dim: ")
    # print(model.operator_node(21).input["Bias"].variable_type[0].tensor.shape.dim)
    # print("*" * 20)
    # print("print all_tensors: ")
    # model.print_all_tensors()
    # print("*" * 20)
    # print("print tensor_name conv2d_49.b_0: ")
    # print(model.tensor_val("conv2d_49.b_0"))
    # print("*" * 20)
    # print("print var conv2d_49.w_0")
    # print(model.variable_type("conv2d_49.w_0"))
    # print("-" * 20)

    if args.save_dir is None:
        args.save_dir = "paddle_model"
        print("will save paddle in: ", args.save_dir)
    model.convert_to_paddle_model(args.save_dir)
