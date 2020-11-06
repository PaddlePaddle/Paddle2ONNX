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
import six
from paddle2onnx.constant.op_mapping_status import *


def get_max_support_version(versions, opset_version):
    max_version = -1
    for vs in sorted(versions):
        if vs <= opset_version:
            max_version = vs
    return max_version


def register_op_mapper(paddle_op, mapper_obj):
    paddle_op_list = []

    if isinstance(paddle_op, six.string_types):
        paddle_op_list.append(paddle_op)
    elif isinstance(paddle_op, list):
        paddle_op_list = paddle_op
    else:
        raise ValueError('paddle_op must be List or string, but got type {}.'.
                         format(type(paddle_op)))

    if not isinstance(mapper_obj, six.class_types):
        raise ValueError('mapper_obj must be Class, but got type {}.'.format(
            type(mapper_obj)))

    valid_register_func = 0
    for k, v in inspect.getmembers(mapper_obj, inspect.ismethod):
        if k.startswith("opset_"):
            version = int(k.replace("opset_", ""))
            if version > 13 or version < 1:
                raise Exception(
                    'the specific method of operator mapper must be named opset_[number](1<=number<=13), such as opset_9, but got {}.'.
                    format(k))
            valid_register_func += 1

    if valid_register_func == 0:
        raise Exception(
            'the specific method of operator mapper must be classmethod, which named opset_[number](1<=number<=13), such as opset_9, but none achieved.'
        )

    mapper = OpMapper(paddle_op_list)
    mapper(mapper_obj)


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
    def mapping(graph, node):
        try:
            opsets = OpMapper.OPSETS[node.type]
            versions = list(opsets.keys())
            convert_version = get_max_support_version(versions,
                                                      graph.opset_version)
            mapper_func, kw = opsets[convert_version]
            mapper_func(graph, node, **kw)
            return OP_MAPPING_SUCCESSED
        except:
            raise Exception(
                "Error happened when mapping node ['{}'] to onnx, which op_type is '{}' with inputs: {} and outputs: {}\n".
                format(node.layer_name, node.type, node.inputs, node.outputs))

    @staticmethod
    def check_support_status(paddle_graph, opset_version):
        op_mapping_status = {
            OP_MAPPING_NO_REGISTER: [],
            OP_MAPPING_NO_VERSION: [],
        }
        for name, node in list(paddle_graph.node_map.items()):
            if node.type not in OpMapper.OPSETS:
                op_mapping_status[OP_MAPPING_NO_REGISTER].append(node)
            else:
                opsets = OpMapper.OPSETS[node.type]
                versions = list(opsets.keys())
                convert_version = get_max_support_version(versions,
                                                          opset_version)
                if convert_version == -1:
                    op_mapping_status[OP_MAPPING_NO_VERSION].append(node)

        if len(op_mapping_status[OP_MAPPING_NO_REGISTER]) > 0:
            unsupported_op_types = set([
                node.type for node in op_mapping_status[OP_MAPPING_NO_REGISTER]
            ])
            error_info = "\nThere's {} ops are not supported yet\n".format(
                len(unsupported_op_types))
            for op_type in unsupported_op_types:
                error_info += "=========== {} ===========\n".format(op_type)
            raise NotImplementedError(error_info)

        if len(op_mapping_status[OP_MAPPING_NO_VERSION]) > 0:
            unsupported_op_types = set([
                node.type for node in op_mapping_status[OP_MAPPING_NO_VERSION]
            ])
            error_info = "\nThere's {} ops are not supported in opset_version {}, please try other opset versions\n".format(
                len(unsupported_op_types), self.opset_version)

            for op_type in unsupported_op_types:
                error_info += "=========== {} ===========\n".format(op_type)
            raise NotImplementedError(error_info)
