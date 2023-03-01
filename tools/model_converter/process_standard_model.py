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
        for graph in self.prog.graphs:
            for var in graph.vars:
                if var.name in self.params2val_dict:
                    shape = var.type.lod_tensor.tensor.dims
                    self.params2val_dict[var.name] = self.params2val_dict[
                        var.name].reshape(shape)

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
            for var in paddle_model.blocks[0].vars:
                if var.name == name:
                    dims = var.type.lod_tensor.tensor.dims
                    val = val.reshape(dims)
                    break
            shape = val.shape
            if len(shape) == 0:
                shape = [1]
            numpy.array([0], dtype='int32').tofile(fp)
            numpy.array([0], dtype='int64').tofile(fp)
            numpy.array([0], dtype='int32').tofile(fp)
            tensor_desc = framework_pb2.VarType.TensorDesc()
            tensor_desc.data_type = framework_pb2.VarType.FP32
            tensor_desc.dims.extend(shape)
            desc_size = tensor_desc.ByteSize()
            numpy.array([desc_size], dtype='int32').tofile(fp)
            fp.write(tensor_desc.SerializeToString())
            val.tofile(fp)
        fp.close()
        print("paddle params saved in: ", params_save_path)

    def convert_model(self, save_dir):
        paddle_model = framework_pb2.ProgramDesc()
        for graph in self.prog.graphs:
            block = paddle_model.blocks.add()
            block.idx = graph.id
            block.parent_idx = graph.parent_idx
            for op in graph.ops:
                operator = helper.make_paddle_operator(op)
                block.ops.append(operator)
            for var in graph.vars:
                var_proto = framework_pb2.VarDesc.FromString(
                    var.SerializeToString())
                block.vars.append(var_proto)

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

    def print_graph(self):
        print(self.prog.graphs)

    def print_node(self, node_index=0, block_index=0):
        Assert(block_index >= 0 and block_index < len(self.prog.graphs),
               "block_idex must be in range [0, " + str(len(self.prog.graphs)) +
               "]")
        Assert(node_index >= 0 and
               node_index < len(self.prog.graphs[block_index].ops),
               "node_index must be in range [0, " +
               str(len(self.prog.graphs[block_index].ops)) + "]")
        print(str(self.prog.graphs[block_index].ops[node_index]))

    def print_var(self, var):
        if isinstance(var, six.string_types):
            vars_str = None
            for vars in self.prog.graphs[0].vars:
                if vars.name == var:
                    vars_str = str(vars)
            if vars_str is None:
                print("Input var is not found: ", var)
            else:
                print(vars_str)

        elif isinstance(var, int):
            Assert(var >= 0 and var < len(self.prog.graphs[0].vars),
                   "tensor must be in range [0, " +
                   str(len(self.prog.graphs[0].vars)) + "]")
            print(str(self.prog.graphs[0].vars[var]))
        else:
            Assert(False, "Please inter a weight name or weight index")

    def print_all_tensors(self):
        for layer_name, param_names in self.layer2params_dict.items():
            print(layer_name)
            for param in param_names:
                print("   ", param, self.params2val_dict[param].shape)

    def print_tensor(self, tensor):
        if isinstance(tensor, six.string_types):
            if tensor in self.params2val_dict:
                print(self.params2val_dict[tensor])
            else:
                print("Input tensor is not found: ", tensor)
        elif isinstance(tensor, int):
            Assert(tensor >= 0 and tensor < len(self.params2val_dict.keys()),
                   "tensor must be in range [0, " +
                   str(len(self.params2val_dict.keys())) + "]")
            tensor_name = self.all_var_names[tensor]
            print("temsor name: ", tensor_name)
            print(self.params2val_dict[tensor_name])
        else:
            Assert(False, "Please inter a weight name or weight index")


if __name__ == '__main__':
    args = parse_arguments()
    paddle.set_device("cpu")
    model = StandardModel(args.standard_model)
    print("*" * 20)
    print("print graph: ")
    model.print_graph()
    print("*" * 20)
    print("print contributors: ")
    print(model.prog.contributors)
    print("*" * 20)
    print("print node_index 21: ")
    print(model.print_node(21))
    print("*" * 20)
    print("print all_tensors: ")
    model.print_all_tensors()
    print("*" * 20)
    print("print tensor_name conv2d_49.b_0: ")
    model.print_tensor("conv2d_49.b_0")
    print("*" * 20)
    print("print var conv2d_49.b_0")
    model.print_var("conv2d_49.b_0")
    print("-" * 20)
    if args.save_dir is None:
        args.save_dir = "paddle_model"
        print("will save paddle in: ", args.save_dir)
    model.convert_to_paddle_model(args.save_dir)
