#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

from paddle2onnx.legacy.passes import PassManager
from paddle2onnx.utils import logging
from paddle2onnx.legacy.passes.quantize_helper import add_missing_quantize_ops_for_tensorrt, new_type_quantize_post_process, remove_all_quantize_ops_and_save_max_range_file, add_missing_quantize_ops_for_onnxruntime, delete_redundant_quantize_ops, add_shortcut_quantize_ops


@PassManager('quantize_model_process_pass')
class QuantizeModelProcessPass(object):
    @classmethod
    def tensorrt_deploy_model(cls, graph):
        graph = delete_redundant_quantize_ops(graph)
        graph = add_missing_quantize_ops_for_tensorrt(graph)
        # graph = add_shortcut_quantize_ops(graph)
        return graph

    @classmethod
    def onnxruntime_deploy_model(cls, graph):
        if graph.quantize_model_mode in ["static", "dynamic"]:
            graph = add_missing_quantize_ops_for_onnxruntime(graph)
            return graph
        elif graph.quantize_model_mode in ["new_type"]:
            graph = new_type_quantize_post_process(graph)
            return graph

    @classmethod
    def other_deploy_model(cls, graph):
        graph = remove_all_quantize_ops_and_save_max_range_file(graph)
        return graph

    @classmethod
    def run_pass(cls, onnx_graph):
        if onnx_graph.quantize_model_mode in ["float"]:
            return onnx_graph
        if onnx_graph.deploy_backend in ["tensorrt"]:
            onnx_graph = cls.tensorrt_deploy_model(onnx_graph)
        elif onnx_graph.deploy_backend in ["onnxruntime"]:
            onnx_graph = cls.onnxruntime_deploy_model(onnx_graph)
        else:
            onnx_graph = cls.other_deploy_model(onnx_graph)

        return onnx_graph
