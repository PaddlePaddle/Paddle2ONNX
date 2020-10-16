# Copyright (c) 2020  PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import absolute_import

import inspect
from paddle2onnx.constant.op_mapping_status import *


def get_max_support_version(versions, opset_version):
    max_version = -1
    for vs in sorted(versions):
        if vs <= opset_version:
            max_version = vs
    return max_version


class OpMapper(object):
    OPSETS = {}

    def __init__(self, paddle_op, **kwargs):
        if not isinstance(paddle_op, list):
            paddle_op = [paddle_op]
        self.paddle_op = paddle_op
        self.kwargs = kwargs

    def __call__(self, cls):
        for k, v in inspect.getmembers(cls, inspect.ismethod):
            if k.startswith("opset_"):
                version = int(k.replace("opset_", ""))
                for op in self.paddle_op:
                    if op not in OpMapper.OPSETS:
                        OpMapper.OPSETS[op] = {}
                    opset_dict = OpMapper.OPSETS[op]
                    opset_dict[version] = (v, self.kwargs)

    @staticmethod
    def mapping(graph, node, opset_version):
        if node.type not in OpMapper.OPSETS:
            return OP_MAPPING_NO_REGISTER
        opsets = OpMapper.OPSETS[node.type]
        versions = list(opsets.keys())
        convert_version = get_max_support_version(versions, opset_version)
        if convert_version == -1:
            return OP_MAPPING_NO_VERSION
        mapper_func, kw = opsets[convert_version]
        try:
            mapper_func(graph, node, **kw)
            # mapped node should be remove from graph 
            graph.remove_node(node)
            return OP_MAPPING_SUCCESSED
        except:
            raise Exception(
                "Error happened when mapping node ['{}'] to onnx, which op_type is '{}' with inputs: {} and outputs: {}\n".
                format(node.layer_name, node.type, node.inputs, node.outputs))
