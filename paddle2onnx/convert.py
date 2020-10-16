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
import onnx
import paddle
import numpy as np
from paddle.nn import Layer
from paddle.fluid import core
from paddle.fluid.framework import Variable
from paddle.fluid.dygraph.dygraph_to_static import program_translator
from paddle.fluid import dygraph
from paddle2onnx.constant import PRODUCER
from paddle2onnx.graph import graph_to_onnx, build_graph


def prepend_feed_ops(inference_program,
                     feed_target_names,
                     feed_holder_name='feed'):
    if len(feed_target_names) == 0:
        return

    global_block = inference_program.global_block()
    feed_var = global_block.create_var(
        name=feed_holder_name,
        type=core.VarDesc.VarType.FEED_MINIBATCH,
        persistable=True)

    for i, name in enumerate(feed_target_names):
        if not global_block.has_var(name):
            raise ValueError(
                "The feeded_var_names[{i}]: '{name}' doesn't exist in pruned inference program. "
                "Please check whether '{name}' is a valid feed_var name, or remove it from feeded_var_names "
                "if '{name}' is not involved in the target_vars calculation.".
                format(
                    i=i, name=name))
        out = global_block.var(name)
        global_block._prepend_op(
            type='feed',
            inputs={'X': [feed_var]},
            outputs={'Out': [out]},
            attrs={'col': i})


def append_fetch_ops(inference_program,
                     fetch_target_names,
                     fetch_holder_name='fetch'):
    global_block = inference_program.global_block()
    fetch_var = global_block.create_var(
        name=fetch_holder_name,
        type=core.VarDesc.VarType.FETCH_LIST,
        persistable=True)

    for i, name in enumerate(fetch_target_names):
        global_block.append_op(
            type='fetch',
            inputs={'X': [name]},
            outputs={'Out': [fetch_var]},
            attrs={'col': i})


def prune_input_output(concrete_program, input_spec, output_spec):
    feeded_vars, feeded_var_names = get_inout_spec(concrete_program.inputs,
                                                   input_spec, True)
    target_vars = get_inout_spec(concrete_program.outputs, output_spec)
    main_program = concrete_program.main_program.clone()
    global_block = main_program.global_block()
    need_to_remove_op_index = []

    for i, op in enumerate(global_block.ops):
        op.desc.set_is_target(False)
        if op.type == "feed" or op.type == "fetch":
            need_to_remove_op_index.append(i)

    for index in need_to_remove_op_index[::-1]:
        global_block._remove_op(index)

    main_program.desc.flush()

    main_program = main_program._prune_with_input(
        feeded_var_names=feeded_var_names, targets=target_vars)
    main_program = main_program._inference_optimize(prune_read_op=True)
    fetch_var_names = [v.name for v in target_vars]

    prepend_feed_ops(main_program, feeded_var_names)
    append_fetch_ops(main_program, fetch_var_names)

    concrete_program.outputs = tuple(target_vars)
    concrete_program.inputs = tuple(feeded_vars)
    concrete_program.main_program = main_program

    return concrete_program


@dygraph.base.switch_to_static_graph
def get_concrete_program(layer):
    paddle.jit.set_verbosity(0)
    if isinstance(layer, Layer):
        if isinstance(layer.forward, program_translator.StaticFunction):
            return layer.forward.concrete_program
        else:
            raise TypeError(
                "The foward of layer should be StaticFunction, but received forward type is %s."
                % type(layer.forward))
    elif isinstance(layer, program_translator.StaticFunction):
        return layer.concrete_program
    else:
        raise TypeError(
            "The input Layer should be 'Layer', but received  type is %s." %
            type(layer))


def get_inout_spec(all_vars, target_vars, return_name=False):
    result_list = []
    valid_var_dict = {}
    valid_vars = [var for var in all_vars if isinstance(var, Variable)]
    for var in valid_vars:
        valid_var_dict[var.name] = var
    if target_vars is not None:
        for i, var in enumerate(target_vars):
            # check target var whether exists
            if var.name not in valid_var_dict:
                raise RuntimeError("The variable to feed/fetch are not exist.")
            result_list.append(valid_var_dict[var.name])
    else:
        result_list = valid_vars
    if return_name:
        return result_list, [var.name for var in result_list]
    return result_list


@dygraph.base.switch_to_static_graph
def build_graph_from_dygraph(layer, input_spec=None, output_spec=None):
    if isinstance(layer, dygraph.TranslatedLayer):
        program = layer.program()
        parameters_dict = {}
        pruned_vars = program.global_block().vars
        for param in layer.parameters():
            if param.name.endswith('feed') or param.name.endswith('fetch'):
                continue
            if not param.persistable:
                continue
            if param.name in pruned_vars:
                parameters_dict[param.name] = {
                    'data': np.array(param.value().get_tensor()),
                    'dtype': param.dtype,
                    'shape': param.shape
                }
        graph = build_graph(program, parameters_dict,
                            layer._input_spec(), layer._output_spec())
        return graph
    elif isinstance(layer, Layer):

        concrete_program = get_concrete_program(layer)

        concrete_program = prune_input_output(concrete_program, input_spec,
                                              output_spec)
        program = concrete_program.main_program

        parameters_dict = {}
        pruned_vars = program.global_block().vars
        for param in concrete_program.parameters:
            if param.name.endswith('feed') or param.name.endswith('fetch'):
                continue
            if not param.persistable:
                continue
            if param.name in pruned_vars:
                parameters_dict[param.name] = {
                    'data': np.array(param.value().get_tensor()),
                    'dtype': param.dtype,
                    'shape': param.shape
                }
        graph = build_graph(program, parameters_dict)
        return graph
    else:
        raise TypeError(
            "The input Layer should be 'Layer' or 'TranslatedLayer', but received  type is %s."
            % type(layer))


def convert_dygraph_to_onnx(layer,
                            save_dir,
                            input_spec=None,
                            opset_version=9,
                            **kwargs):
    if not isinstance(layer, Layer):
        raise TypeError(
            "the input 'layer' of paddle.onnx.export should be 'Layer', 'TranslatedLayer', but received type is %s."
            % type(layer))

    inner_input_spec = None
    if input_spec is not None:
        if not isinstance(input_spec, list):
            raise TypeError(
                "The input input_spec should be 'list', but received type is %s."
                % type(input_spec))
        inner_input_spec = []
        for var in input_spec:
            if isinstance(var, paddle.static.InputSpec):
                inner_input_spec.append(var)
            elif isinstance(var, (core.VarBase, Variable)):
                inner_input_spec.append(
                    paddle.static.InputSpec.from_tensor(var))
            else:
                raise TypeError(
                    "The element in input_spec list should be 'Variable' or `paddle.static.InputSpec`, but received element's type is %s."
                    % type(var))

    output_spec = None
    if 'output_spec' in kwargs:
        output_spec = kwargs['output_spec']
        if not isinstance(output_spec, list):
            raise TypeError(
                "The output_spec should be 'list', but received type is %s." %
                type(output_spec))
            for var in output_spec:
                if not isinstance(var, core.VarBase):
                    raise TypeError(
                        "The element in output_spec list should be 'Variable', but received element's type is %s."
                        % type(var))
    verbose = False
    if 'verbose' in kwargs:
        if isinstance(kwargs['verbose'], bool):
            verbose = kwargs['verbose']
        else:
            raise TypeError(
                "The verbose should be 'bool', but received type is %s." %
                type(kwargs['verbose']))

    graph = build_graph_from_dygraph(layer, inner_input_spec, output_spec)

    onnx_graphs = graph_to_onnx(graph, opset_version, verbose=verbose)

    onnx_graph = onnx_graphs[0]

    opset_imports = [onnx.helper.make_opsetid("", opset_version)]
    onnx_model = onnx.helper.make_model(
        onnx_graph, producer_name=PRODUCER, opset_imports=opset_imports)

    onnx.checker.check_model(onnx_model)

    path, _ = os.path.split(save_dir)
    if path != '' and not os.path.isdir(path):
        os.makedirs(path)
    with open(save_dir, 'wb') as f:
        f.write(onnx_model.SerializeToString())

    print("ONNX model saved in {}".format(save_dir))
