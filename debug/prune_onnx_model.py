# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

import onnx
from onnx import helper, TensorProto
import os
import math
import logging

logger = logging.getLogger("p2o-logger")


def prune_onnx_model(
    onnx_model_file,
    target_node_name="p2o.print",
    target_dims=[1],
    target_dtype="float32",
):
    dtype_map = {
        "bool": (TensorProto.BOOL, bool),
        "float32": (TensorProto.FLOAT, float),
        "float64": (TensorProto.DOUBLE, float),
        "int32": (TensorProto.INT32, int),
        "int64": (TensorProto.INT64, int),
    }
    model = onnx.load(onnx_model_file)
    onnx.checker.check_model(model)
    loop_body = None
    loop_node = None
    target_node_output = None
    for n in model.graph.node:
        if n.op_type == "Loop":
            loop_node = n
            loop_body = n.attribute[0].g
            for sub_n in loop_body.node:
                if sub_n.name.startswith(target_node_name):
                    target_node_output = sub_n.output[0]
                    break
    if target_node_output is None:
        raise ValueError(f"Cannot find target node '{target_node_name}' in Loop.")
    first_iter_initial_name = "first_iter_initial"
    first_iter_initial = helper.make_tensor(
        name=first_iter_initial_name,
        data_type=dtype_map[target_dtype][0],
        dims=target_dims if len(target_dims) > 0 else (),
        vals=(
            [dtype_map[target_dtype][1](0) for _ in range(math.prod(target_dims))]
            if len(target_dims) > 0
            else [dtype_map[target_dtype][1](0)]
        ),
    )
    model.graph.initializer.append(first_iter_initial)

    loop_node.input.append(first_iter_initial_name)

    first_iter_output = helper.make_tensor_value_info(
        "first_iter_output", dtype_map[target_dtype][0], target_dims
    )
    loop_body.input.append(first_iter_output)

    zero_const_tensor = helper.make_tensor(
        name="zero_const",
        data_type=TensorProto.INT64,
        dims=(),
        vals=[0],
    )
    model.graph.initializer.append(zero_const_tensor)
    first_iter_cond = helper.make_node(
        "Equal",
        inputs=[loop_body.input[0].name, zero_const_tensor.name],
        outputs=["first_iter_cond"],
    )

    if_node = helper.make_node(
        "If",
        inputs=["first_iter_cond"],
        outputs=["first_iter_output_next"],
        then_branch=helper.make_graph(
            [
                helper.make_node(
                    "Identity", [target_node_output], ["first_iter_output_next"]
                )
            ],
            "then_branch",
            [],
            [
                helper.make_tensor_value_info(
                    "first_iter_output_next", dtype_map[target_dtype][0], target_dims
                )
            ],
        ),
        else_branch=helper.make_graph(
            [
                helper.make_node(
                    "Identity", [first_iter_output.name], ["first_iter_output_next"]
                )
            ],
            "else_branch",
            [],
            [
                helper.make_tensor_value_info(
                    "first_iter_output_next", dtype_map[target_dtype][0], target_dims
                )
            ],
        ),
    )

    loop_body.node.extend([first_iter_cond, if_node])

    loop_body.output.append(
        helper.make_tensor_value_info(
            "first_iter_output_next", dtype_map[target_dtype][0], target_dims
        )
    )
    loop_node.output.append("first_iter_output_next")
    model.graph.output.append(
        helper.make_tensor_value_info(
            "first_iter_output_next", dtype_map[target_dtype][0], target_dims
        )
    )

    output_model_file = os.path.basename(onnx_model_file) + "_modified.onnx"

    onnx.save(model, output_model_file)
    logger.info("Modified onnx model saved to %s", output_model_file)
    return output_model_file
