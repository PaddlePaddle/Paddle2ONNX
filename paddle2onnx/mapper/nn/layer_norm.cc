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
#include <string>
#include <vector>

namespace paddle2onnx {
REGISTER_MAPPER(layer_norm, LayerNormMapper)

void LayerNormMapper::Opset7(OnnxHelper* helper) {
  std::vector<TensorInfo> input_info =
      parser_->GetOpInput(block_idx_, op_idx_, "X");
  std::vector<TensorInfo> output_info =
      parser_->GetOpOutput(block_idx_, op_idx_, "Y");

  std::vector<int64_t> input_shape = input_info[0].shape;
  std::vector<int64_t> axes;
  for (auto i = begin_norm_axis_; i < input_shape.size(); i++) {
    axes.push_back(i);
  }

  std::string epsilon_node =
      helper->MakeConstant({1}, GetOnnxDtype(input_info[0].dtype), epsilon_)
          ->output(0);
  auto two_node =
      helper->MakeConstant({1}, GetOnnxDtype(input_info[0].dtype), 2.0);

  auto mean_node = helper->MakeNode("ReduceMean", {input_info[0].name});
  AddAttribute(mean_node, "axes", axes);

  auto numerator_node =
      helper->MakeNode("Sub", {input_info[0].name, mean_node->output(0)});
  auto pow_num_node =
      helper->MakeNode("Pow", {numerator_node->output(0), two_node->output(0)});

  auto variance_node =
      helper->MakeNode("ReduceMean", {pow_num_node->output(0)});
  AddAttribute(variance_node, "axes", axes);

  auto add_eps_node =
      helper->MakeNode("Add", {variance_node->output(0), epsilon_node});

  auto denominator_node = helper->MakeNode("Sqrt", {add_eps_node->output(0)});

  auto ipt_shape_node = helper->MakeNode("Shape", {input_info[0].name});
  std::vector<int64_t> slice_axes = {0};
  std::vector<int64_t> start = {
      static_cast<int64_t>(input_shape.size() - axes.size())};
  std::vector<int64_t> end = {static_cast<int64_t>(input_shape.size())};
  std::string weight_shape_node =
      helper->Slice(ipt_shape_node->output(0), slice_axes, start, end);

  bool has_input_Bias = parser_->OpHasInput(block_idx_, op_idx_, "Bias");
  bool has_input_Scale = parser_->OpHasInput(block_idx_, op_idx_, "Scale");

  if (has_input_Bias && has_input_Scale) {
    std::vector<TensorInfo> scale_info =
        parser_->GetOpInput(block_idx_, op_idx_, "Scale");
    std::vector<TensorInfo> bias_info =
        parser_->GetOpInput(block_idx_, op_idx_, "Bias");
    auto scale_node =
        helper->MakeNode("Reshape", {scale_info[0].name, weight_shape_node});
    auto bias_node =
        helper->MakeNode("Reshape", {bias_info[0].name, weight_shape_node});
    auto layer_norm_pre_node = helper->MakeNode(
        "Div", {numerator_node->output(0), denominator_node->output(0)});
    auto layer_norm_node = helper->MakeNode(
        "Mul", {layer_norm_pre_node->output(0), scale_node->output(0)});
    helper->MakeNode("Add", {layer_norm_node->output(0), bias_node->output(0)},
                     {output_info[0].name});
    return;
  }
  if (has_input_Bias) {
    std::vector<TensorInfo> bias_info =
        parser_->GetOpInput(block_idx_, op_idx_, "Bias");
    auto bias_node =
        helper->MakeNode("Reshape", {bias_info[0].name, weight_shape_node});
    auto layer_norm_node = helper->MakeNode(
        "Div", {numerator_node->output(0), denominator_node->output(0)});
    helper->MakeNode("Add", {layer_norm_node->output(0), bias_node->output(0)},
                     {output_info[0].name});
    return;
  }
  if (has_input_Scale) {
    std::vector<TensorInfo> scale_info =
        parser_->GetOpInput(block_idx_, op_idx_, "Scale");
    auto scale_node =
        helper->MakeNode("Reshape", {scale_info[0].name, weight_shape_node});
    auto layer_norm_node = helper->MakeNode(
        "Div", {numerator_node->output(0), denominator_node->output(0)});
    helper->MakeNode("Mul", {layer_norm_node->output(0), scale_node->output(0)},
                     {output_info[0].name});
    return;
  }
  helper->MakeNode("Div",
                   {numerator_node->output(0), denominator_node->output(0)},
                   {output_info[0].name});
}

}  // namespace paddle2onnx
