# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from six import text_type as _text_type
import os

__version__ = "0.9.3"

def export(model_filename,
           params_filename="",
           save_file=None,
           opset_version=9,
           auto_upgrade_opset=True,
           verbose=True,
           enable_onnx_checker=True,
           enable_experimental_op=True,
           enable_optimize=True):
    import paddle2onnx.paddle2onnx_cpp2py_export as c_p2o
    onnx_proto_str = c_p2o.export(
        model_filename, params_filename, opset_version, auto_upgrade_opset,
        verbose, False, enable_experimental_op, enable_optimize)
    if enable_onnx_chekcer:
        import onnx
        model = onnx.ModelProto()
        model.ParseFromString(onnx_proto_str)
        onnx.checker.check_model(model)
    if save_file is not None:
        with open(save_file, "wb") as f:
            f.write(onnx_proto_str)
    else:
        return onnx_proto_str


# internal api, not recommend to use
def dygraph2onnx(layer, save_file, input_spec, opset_version=9, **configs):
    import os
    import paddle
    dirname = os.path.split(save_file)[0]
    paddle.jit.save(layer, os.path.join(dirname, "model"), input_spec)
    auto_upgrade_opset = False
    if 'auto_upgrade_opset' in configs:
        if isinstance(configs['auto_upgrade_opset'], bool):
            auto_upgrade_opset = configs['auto_upgrade_opset']
        else:
            raise TypeError(
                "The auto_upgrade_opset should be 'bool', but received type is %s."
                % type(configs['auto_upgrade_opset']))

    get_op_list = False
    if 'get_op_list' in configs:
        if isinstance(configs['get_op_list'], bool):
            get_op_list = configs['get_op_list']
        else:
            raise TypeError(
                "The get_op_list should be 'bool', but received type is %s." %
                type(configs['get_op_list']))

    model_file = os.path.join(dirname, 'model.pdmodel')
    params_file = os.path.join(dirname, 'model.pdiparams')
    if not os.path.exists(params_file):
        params_file = ""
    if get_op_list:
        op_list = c_p2o.get_graph_op_list(model_file, params_file)
        return op_list

    export(
        model_file,
        params_file,
        save_file=save_file,
        opset_version=opset_version,
        auto_upgrade_opset=auto_upgrade_opset)

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', "-m", required=True, type=_text_type, helper='The directory path of the paddlepaddle model.')
    parser.add_argument('--model_filename', "-mf", required=True, type=_text_type, help='The model file name under the directory designated by --model_dir.')
    parser.add_argument('--params_filename', "-pf", required=True, help='The parameter file name under the directory designated by --model_dir.')
    parser.add_argument('--save_file', "-s", required=True, help='Path to dump onnx model file.')
    parser.add_argument('--opset_version', "-ov", type=int, default=9, help="Configure the ONNX Opset version, 7-15 are stably supported. Default 9.")
    parser.add_argument("--auto_upgrade_opset", type=ast.literal_eval, default=True, help="If auto upgrade the opset_version for successfuly conversion. Default True.")
    parser.add_argument("--enable_onnx_checker", type=ast.literal_eval, default=True, help="If check the converted ONNX model is illegal or not. Default True.")
    parser.add_argument("--version", "-v", action="store_true", default=False, help="Get version of Paddle2ONNX.")
    parser.add_argument("--enable_develop_paddle2onnx", action="store_true", default=False, help="[Experimental] This is a new flag, by setting this, we will use new developed Paddle2ONNX. This flag will be DEPRECATED after the new developed Paddle2ONNX is stable enough to replace the old version.")
    parser.add_argument("--enable_onnx_optimize", action="store_true", default=True, help="If optimize the converted onnx model, this flag is working only --enable_develop_paddle2onnx=True. Default True")
    parser.add_argument("--input_shape_dict", "-isd", type=_text_type, default="None", help="Define input shapes, e.g --input_shape_dict=\"{'image':[1, 3, 608, 608]}\" or" \"--input_shape_dict=\"{'image':[1, 3, 608, 608], 'im_shape': [1, 2], 'scale_factor': [1, 2]}\", this flag is discarded while --enable_new_version=True, and the flag will be DEPRECATED after the next version.")
    return parser.parse_args()

def main():
    parser = arg_parser()
    args = parser.parse_args()
    if args.version:
        print("Paddle2ONNX v{}".format(__version__))
    return

    if args.enable_develop_paddle2onnx:
        print("[INFO] You are using a new version of Paddle2ONNX, if there's any problem, please report to us in https://github.com/PaddlePaddle/Paddle2ONNX.git"
        if args.input_shape_dict != "None":
            print("[WARN] --input_shape_dict is not working while --enable_new_version=True.")
        export(model_filename=os.path.join(args.model_dir, args.model_filename),
               params_filename=os.path.join(args.model_dir, args.params_filename),
               save_file=args.save_file,
               opset_version=args.opset_version,
               auto_upgrade_opset=args.auto_upgrade_opset,
               verbose=True,
               enable_onnx_checker=args.enable_onnx_checker,
               enable_experimental_op=True,
               enable_optimize=args.enable_onnx_optimize):
    else:
        from legacy import paddle2onnx
