# Copyright (c) 2022  PaddlePaddle Authors. All Rights Reserved.
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

from paddle2onnx.utils import logging

def export_onnx(paddle_graph,
                save_file,
                opset_version=9,
                enable_onnx_checker=False,
                operator_export_type="ONNX",
                verbose=False,
                auto_update_opset=True,
                output_names=None):
    from paddle2onnx.legacy.convert import export_onnx
    return export_onnx(paddle_graph, save_file, opset_version, opset_version, enable_onnx_checker, operator_export_type, verbose, auto_update_opset, output_names)

def dygraph2onnx(layer, save_file, input_spec=None, opset_version=9, **configs):
    from paddle2onnx.legacy.convert import dygraph2onnx
    return dygraph2onnx(layer, save_file, input_spec, opset_version, **configs)

def program2onnx(program,
                 scope,
                 save_file,
                 feed_var_names=None,
                 target_vars=None,
                 opset_version=9,
                 enable_onnx_checker=False,
                 operator_export_type="ONNX",
                 auto_update_opset=True,
                 **configs):
    from paddle2onnx.legacy.convert import program2onnx
    return program2onnx(program, scope, save_file, feed_var_names, target_vars, opset_version, enable_onnx_checker, operator_export_type, auto_update_opset, **configs)
