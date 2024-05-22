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

#include "paddle2onnx/mapper/nn/layer_norm.h"

#include <cmath>
#include <string>
#include <vector>
#include <array>

namespace paddle2onnx {
REGISTER_MAPPER(layer_norm, LayerNormMapper)

void LayerNormMapper::Opset17() {
  auto input_info = GetInput("X");
  auto output_info = GetOutput("Y");

  constexpr std::array<P2ODataType, 3> T = {P2ODataType::FP16, P2ODataType::FP32, P2ODataType::FP64};

  auto input_name = input_info[0].name;
  auto input_type = input_info[0].dtype;
  auto input_shape = input_info[0].shape;
  if (std::find(T.begin(), T.end(), input_type) == T.end()) {
    input_name = helper_->AutoCast(input_name, input_info[0].dtype, P2ODataType::FP32);
    input_type = P2ODataType::FP32;
  }

  bool has_input_Bias = HasInput("Bias");
  bool has_input_Scale = HasInput("Scale");
  if (has_input_Bias && has_input_Scale) {
    auto scale_info = GetInput("Scale");
    auto scale_name = scale_info[0].name;
    auto scale_type = scale_info[0].dtype;
    if (std::find(T.begin(), T.end(), scale_type) == T.end()) {
      scale_name = helper_->AutoCast(scale_name, scale_type, P2ODataType::FP32);
      scale_type = P2ODataType::FP32;
    }

    auto bias_info = GetInput("Bias");
    auto bias_name = bias_info[0].name;
    auto bias_type = bias_info[0].dtype;
    if (std::find(T.begin(), T.end(), bias_type) == T.end()) {
      bias_name = helper_->AutoCast(bias_name, bias_type, P2ODataType::FP32);
      bias_type = P2ODataType::FP32;
    }

    auto layer_norm_node = helper_->MakeNode(
      "LayerNormalization",
      {input_name, scale_name, bias_name},
      {output_info[0].name});
    AddAttribute(layer_norm_node, "axis", begin_norm_axis_);
    AddAttribute(layer_norm_node, "epsilon", epsilon_);
    return;
  }

  if (has_input_Scale) {
    auto scale_info = GetInput("Scale");
    auto scale_name = scale_info[0].name;
    auto scale_type = scale_info[0].dtype;
    if (std::find(T.begin(), T.end(), scale_type) == T.end()) {
      scale_name = helper_->AutoCast(scale_name, scale_type, P2ODataType::FP32);
      scale_type = P2ODataType::FP32;
    }

    auto layer_norm_node = helper_->MakeNode(
      "LayerNormalization",
      {input_name, scale_name},
      {output_info[0].name});
    AddAttribute(layer_norm_node, "axis", begin_norm_axis_);
    AddAttribute(layer_norm_node, "epsilon", epsilon_);
    return;
  }

  std::vector<int64_t> normalized_shape;
  for (int64_t i = begin_norm_axis_;i < input_shape.size();i++) {
    normalized_shape.emplace_back(input_shape[i]);
  }

  if (has_input_Bias) {
    auto bias_info = GetInput("Bias");
    auto bias_name = bias_info[0].name;
    auto bias_type = bias_info[0].dtype;
    if (std::find(T.begin(), T.end(), bias_type) == T.end()) {
      bias_name = helper_->AutoCast(bias_name, bias_type, P2ODataType::FP32);
      bias_type = P2ODataType::FP32;
    }

    std::string scale_name = helper_->Constant(normalized_shape, GetOnnxDtype(P2ODataType::FP32), static_cast<float>(1.0));
    auto layer_norm_node = helper_->MakeNode(
      "LayerNormalization",
      {input_name, scale_name, bias_name},
      {output_info[0].name});
    AddAttribute(layer_norm_node, "axis", begin_norm_axis_);
    AddAttribute(layer_norm_node, "epsilon", epsilon_);
    return;
  }

  if (!has_input_Bias && !has_input_Scale) {
    std::string scale_name = helper_->Constant(normalized_shape, GetOnnxDtype(P2ODataType::FP32), static_cast<float>(1.0));
    auto layer_norm_node = helper_->MakeNode(
      "LayerNormalization",
      {input_name, scale_name},
      {output_info[0].name});
    AddAttribute(layer_norm_node, "axis", begin_norm_axis_);
    AddAttribute(layer_norm_node, "epsilon", epsilon_);
  }
}

void LayerNormMapper::Opset7() {
  auto input_info = GetInput("X");
  auto output_info = GetOutput("Y");

  std::string input_name = helper_->AutoCast(
    input_info[0].name,
    input_info[0].dtype,
    P2ODataType::FP32);

  std::vector<int64_t> input_shape = input_info[0].shape;
  std::vector<int64_t> axes;
  for (auto i = begin_norm_axis_; i < input_shape.size(); i++) {
    axes.push_back(i);
  }
  if (begin_norm_axis_ == input_shape.size() - 1) {
    axes[0] = -1;
  }

  float epsilon = epsilon_;
  std::string epsilon_node = helper_->Constant({}, GetOnnxDtype(P2ODataType::FP32), epsilon);
  std::string two_node = helper_->Constant({}, GetOnnxDtype(P2ODataType::FP32), float(2.0));

  auto mean_node = helper_->MakeNode("ReduceMean", {input_name});
  AddAttribute(mean_node, "axes", axes);

  auto numerator_node = helper_->MakeNode("Sub", {input_name, mean_node->output(0)});
  auto pow_num_node = helper_->MakeNode("Pow", {numerator_node->output(0), two_node});

  auto variance_node = helper_->MakeNode("ReduceMean", {pow_num_node->output(0)});
  AddAttribute(variance_node, "axes", axes);

  auto add_eps_node = helper_->MakeNode("Add", {variance_node->output(0), epsilon_node});

  auto denominator_node = helper_->MakeNode("Sqrt", {add_eps_node->output(0)});

  auto ipt_shape_node = helper_->MakeNode("Shape", {input_name});
  std::vector<int64_t> slice_axes = {0};
  std::vector<int64_t> start = {
    static_cast<int64_t>(input_shape.size() - axes.size())
  };
  std::vector<int64_t> end = {static_cast<int64_t>(input_shape.size())};
  std::string weight_shape_node =
      helper_->Slice(ipt_shape_node->output(0), slice_axes, start, end);

  bool has_input_Bias = HasInput("Bias");
  bool has_input_Scale = HasInput("Scale");

  if (has_input_Bias && has_input_Scale) {
    auto scale_info = GetInput("Scale");
    auto bias_info = GetInput("Bias");
    std::string scale_name = helper_->AutoCast(
      scale_info[0].name,
      scale_info[0].dtype,
      P2ODataType::FP32);
    std::string bias_name = helper_->AutoCast(
      bias_info[0].name,
      bias_info[0].dtype,
      P2ODataType::FP32);
    std::string scale_node = "";
    std::string bias_node = "";
    if (begin_norm_axis_ == input_shape.size() - 1) {
      scale_node = helper_->Reshape(scale_name, {-1});
      bias_node = helper_->Reshape(bias_name, {-1});
    } else {
      scale_node = helper_->MakeNode("Reshape", {scale_name, weight_shape_node})->output(0);
      bias_node = helper_->MakeNode("Reshape", {bias_name, weight_shape_node})->output(0);
    }
    auto layer_norm_pre_node = helper_->MakeNode(
      "Div",
      {numerator_node->output(0), denominator_node->output(0)});
    auto layer_norm_node = helper_->MakeNode("Mul", {layer_norm_pre_node->output(0), scale_node});
    auto pre_cast_node = helper_->MakeNode("Add", {layer_norm_node->output(0), bias_node});
    helper_->AutoCast(pre_cast_node->output(0),
                      output_info[0].name,
                      P2ODataType::FP32,
                      output_info[0].dtype);
    return;
  }
  if (has_input_Bias) {
    auto bias_info = GetInput("Bias");
    std::string bias_name = helper_->AutoCast(
      bias_info[0].name,
      bias_info[0].dtype,
      P2ODataType::FP32);
    std::string bias_node = "";
    if (begin_norm_axis_ == input_shape.size() - 1) {
      bias_node = helper_->Reshape(bias_name, {-1});
    } else {
      bias_node = helper_->MakeNode("Reshape", {bias_name, weight_shape_node})
          ->output(0);
    }
    auto layer_norm_node = helper_->MakeNode(
      "Div",
      {numerator_node->output(0), denominator_node->output(0)});
    auto pre_cast_node =
        helper_->MakeNode("Add", {layer_norm_node->output(0), bias_node});
    helper_->AutoCast(pre_cast_node->output(0),
                      output_info[0].name,
                      P2ODataType::FP32,
                      output_info[0].dtype);
    return;
  }
  if (has_input_Scale) {
    auto scale_info = GetInput("Scale");
    std::string scale_name = helper_->AutoCast(
      scale_info[0].name,
      scale_info[0].dtype,
      P2ODataType::FP32);
    std::string scale_node = "";
    if (begin_norm_axis_ == input_shape.size() - 1) {
      scale_node = helper_->Reshape(scale_name, {-1});
    } else {
      scale_node = helper_->MakeNode("Reshape", {scale_name, weight_shape_node})
          ->output(0);
    }
    auto layer_norm_node = helper_->MakeNode(
      "Div",
      {numerator_node->output(0), denominator_node->output(0)});
    auto pre_cast_node =
        helper_->MakeNode("Mul", {layer_norm_node->output(0), scale_node});
    helper_->AutoCast(pre_cast_node->output(0),
                      output_info[0].name,
                      P2ODataType::FP32,
                      output_info[0].dtype);
    return;
  }
  auto pre_cast_node = helper_->MakeNode(
    "Div",
    {numerator_node->output(0), denominator_node->output(0)});
  helper_->AutoCast(pre_cast_node->output(0),
                    output_info[0].name,
                    P2ODataType::FP32,
                    output_info[0].dtype);
}
} // namespace paddle2onnx
