// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include "paddle2onnx/mapper/tensor/scale.h"

#include <vector>

namespace paddle2onnx {

REGISTER_MAPPER(scale, ScaleMapper)

void ScaleMapper::Opset7(OnnxHelper* helper) {
  std::vector<TensorInfo> input_info = GetInput("X");
  std::vector<TensorInfo> output_info = GetOutput("Out");
  bool has_scale_tensor = HasInput("ScaleTensor");
  // TODO(yeliang2258): just temporary use Identity
  bool is_scale_1 = ((scale_ - 1.0) < 1e-06 && (scale_ - 1.0) > -1e-06);
  bool is_bias_0 = (bias_ < 1e-06 && bias_ > -1e-06);
  if (!has_scale_tensor && is_scale_1 && is_bias_0) {
    // TODO(yeliang2258): we could add a pass to eleminate all the identity op
    helper->MakeNode("Identity", {input_info[0].name}, {output_info[0].name});
  } else {
    // TODO(yeliang2258): we could add a pass to eleminate the scale is 1 or
    // bias is 0
    int32_t data_type = input_info[0].dtype;
    std::string cast_node = input_info[0].name;
    bool is_int_input = false;
    if (input_info[0].dtype == P2ODataType::INT64 ||
        input_info[0].dtype == P2ODataType::INT32 ||
        input_info[0].dtype == P2ODataType::INT16) {
      std::cerr << " Int type input may bring calculation diff in op "
                << OpType() << "." << std::endl;
      cast_node = helper->AutoCast(input_info[0].name, input_info[0].dtype,
                                   P2ODataType::FP32);
      data_type = P2ODataType::FP32;
      is_int_input = true;
    }

    std::string scale_node;
    if (has_scale_tensor) {
      std::vector<TensorInfo> scale_tensor_info = GetInput("ScaleTensor");
      scale_node = helper->AutoCast(scale_tensor_info[0].name,
                                    scale_tensor_info[0].dtype, data_type);
    } else {
      scale_node =
          helper->Constant({1}, GetOnnxDtype(input_info[0].dtype), scale_);
    }

    std::string bias_node =
        helper->Constant({1}, GetOnnxDtype(input_info[0].dtype), bias_);

    if (bias_after_scale_) {
      auto mul_node = helper->MakeNode("Mul", {cast_node, scale_node});
      if (!is_int_input) {
        helper->MakeNode("Add", {mul_node->output(0), bias_node},
                         {output_info[0].name});
      } else {
        std::string node2 =
            helper->MakeNode("Add", {mul_node->output(0), bias_node})
                ->output(0);
        helper->AutoCast(node2, {output_info[0].name}, data_type,
                         output_info[0].dtype);
      }
    } else {
      auto add_node = helper->MakeNode("Add", {cast_node, bias_node});
      if (!is_int_input) {
        helper->MakeNode("Mul", {add_node->output(0), scale_node},
                         {output_info[0].name});
      } else {
        std::string node2 =
            helper->MakeNode("Mul", {add_node->output(0), scale_node})
                ->output(0);
        helper->AutoCast(node2, {output_info[0].name}, data_type,
                         output_info[0].dtype);
      }
    }
  }
}
}  // namespace paddle2onnx
