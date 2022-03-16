#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import paddle2onnx.paddle2onnx_cpp2py_export as c_p2o
from paddle2onnx import version
import onnx


def export(model_filename,
           params_filename="",
           save_file=None,
           opset_version=9,
           auto_upgrade_opset=True,
           verbose=True,
           enable_onnx_checker=True,
           enable_experimental_op=True):
    onnx_proto = onnx.ModelProto()
    onnx_proto_str = c_p2o.export(model_filename, params_filename,
                                  opset_version, auto_upgrade_opset, verbose,
                                  enable_onnx_checker, enable_experimental_op)
    onnx_proto.ParseFromString(onnx_proto_str)
    if save_file is not None:
        with open(save_file, "wb") as f:
            f.write(onnx_proto.SerializeToString())


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
