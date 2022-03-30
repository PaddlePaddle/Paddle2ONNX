#   Copyright (c) 2020  PaddlePaddle Authors. All Rights Reserved.
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

import os
import numpy as np
import inspect
import six
import paddle
from paddle.fluid.io import _get_valid_program
from paddle.fluid.dygraph.dygraph_to_static.program_translator import ProgramTranslator, StaticFunction
from paddle.fluid.layers.utils import flatten, pack_sequence_as
from collections import OrderedDict
from paddle.fluid import dygraph
from paddle.fluid.dygraph.jit import declarative
from paddle.fluid import core
from paddle.fluid import layers
from paddle.nn import Layer
from paddle.fluid.framework import Block, ParamBase, Program, Variable, Parameter, program_guard
from paddle.fluid.dygraph.layers import Layer

from paddle2onnx.utils import logging
from paddle2onnx.graph.graph_helper import prepend_feed_ops, append_fetch_ops


def _get_input_var_names(inputs, input_spec):
    name_none_error = "The %s's name is None. " \
        "When using jit.save, please set InputSepc's name in " \
        "to_static(input_spec=[]) and jit.save(input_spec=[]) " \
        "and make sure they are consistent."
    name_no_exists_error = "The tensor `%s` does not exists. " \
        "Please make sure the name of InputSpec or example Tensor " \
        "in input_spec is the same as the name of InputSpec in " \
        "`to_static` decorated on the Layer.forward method."
    result_list = []
    input_var_names = [
        var.name for var in flatten(inputs) if isinstance(var, Variable)
    ]
    if input_spec is None:
        # no prune
        return input_var_names
    else:
        # fileter out non-tensor type spec infos.
        input_spec = [
            spec for spec in input_spec
            if isinstance(spec, paddle.static.InputSpec)
        ]

    if len(input_spec) == len(input_var_names):
        # no prune
        result_list = input_var_names
        # if input spec name not in input_var_names, only raise warning
        for spec in input_spec:
            if spec.name is None:
                warnings.warn(name_none_error % spec)
            elif spec.name not in input_var_names:
                warnings.warn(name_no_exists_error % spec.name)
            else:
                # do nothing
                pass
    else:
        # prune
        for spec in input_spec:
            if spec.name is None:
                # name is None, the input_spec only can be InputSpec
                raise ValueError(name_none_error % spec)
            elif spec.name not in input_var_names:
                # the input_spec can be `InputSpec` or `VarBase`
                raise ValueError(name_no_exists_error % spec.name)
            else:
                result_list.append(spec.name)

    return result_list


def _get_output_vars(outputs, output_spec):
    name_no_exists_error = "The tensor `%s` does not exists. " \
        "Please make sure the name of example Tensor " \
        "in configs.output_spec is the output tensor of " \
        "Layer.forward method."
    result_list = []
    output_vars_dict = OrderedDict()
    for var in flatten(outputs):
        if isinstance(var, Variable):
            output_vars_dict[var.name] = var
    if output_spec is None:
        result_list = output_vars_dict.values()
    elif output_spec is not None and len(output_spec) == len(output_vars_dict):
        result_list = output_vars_dict.values()
        for var in output_spec:
            if var.name not in output_vars_dict:
                warnings.warn(name_no_exists_error % var.name)
    else:
        for var in output_spec:
            if var.name not in output_vars_dict:
                raise ValueError(name_no_exists_error % var.name)
            else:
                result_list.append(output_vars_dict[var.name])
    return result_list


def get_program(layer, input_spec, output_spec, **configs):
    layer = paddle.jit.to_static(layer)
    concrete_program = layer.forward.concrete_program_specify_input_spec(
        input_spec=input_spec)
    main_program = concrete_program.main_program
    feeded_var_names = _get_input_var_names(concrete_program.inputs, input_spec)
    target_vars = _get_output_vars(concrete_program.outputs, output_spec)

    return main_program, feeded_var_names, target_vars
