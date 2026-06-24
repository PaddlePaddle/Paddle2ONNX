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

#include "paddle2onnx/mapper/nn/group_norm.h"
#include <cmath>
#include <string>
#include <vector>

namespace paddle2onnx {
REGISTER_MAPPER(group_norm, GroupNormMapper)
REGISTER_PIR_MAPPER(group_norm, GroupNormMapper)

int32_t GroupNormMapper::GetMinOpsetVersion(bool verbose) {
  if (in_pir_mode) {
    return 7;
  }
  auto input_info = GetInput("X");
  if (input_info[0].Rank() != 4) {
    Error() << "Only support 4D-Tensor as input for GroupNorm" << std::endl;
    return -1;
  }
  return 7;
}

void GroupNormMapper::Opset7() {
  std::string input_name, scale_name, bias_name, output_name;
  bool has_scale = false, has_bias = false;

  P2ODataType input_dtype;
  if (in_pir_mode) {
    auto x_info = GetInput(0);
    input_name = x_info[0].name;
    input_dtype = x_info[0].dtype;

    auto s_info = GetInput(1);
    has_scale = (s_info.size() > 0 && s_info[0].name.size() > 0);
    if (has_scale) scale_name = s_info[0].name;

    auto b_info = GetInput(2);
    has_bias = (b_info.size() > 0 && b_info[0].name.size() > 0);
    if (has_bias) bias_name = b_info[0].name;

    auto out_info = GetOutput(0);
    output_name = out_info[0].name;
  } else {
    auto input_info = GetInput("X");
    input_name = input_info[0].name;
    input_dtype = input_info[0].dtype;

    if (HasInput("Scale")) {
      scale_name = GetInput("Scale")[0].name;
      has_scale = true;
    }
    if (HasInput("Bias")) {
      bias_name = GetInput("Bias")[0].name;
      has_bias = true;
    }
    auto output_info = GetOutput("Y");
    output_name = output_info[0].name;
  }

  // PIR mode: 3D input [N, C, L] -> unsqueeze to 4D [N, C, L, 1]
  if (in_pir_mode) {
    input_name = helper_->Unsqueeze(input_name, {3});
  }

  // Reshape: [N, C, H, W] -> [N, groups, -1]
  std::vector<int64_t> shape_val = {0, groups_, -1};
  std::string shape =
      helper_->Constant(GetOnnxDtype(P2ODataType::INT64), shape_val);
  auto reshape_in = helper_->MakeNode("Reshape", {input_name, shape});

  // InstanceNormalization with dummy scale/bias (size=groups)
  // Real scale/bias applied afterwards.
  // Use input dtype to avoid type mismatch when input is FP16/FP64.
  auto onnx_dtype = GetOnnxDtype(input_dtype);
  std::string dummy_scale =
      helper_->Constant(onnx_dtype, std::vector<float>(groups_, 1.0));
  std::string dummy_bias =
      helper_->Constant(onnx_dtype, std::vector<float>(groups_, 0.0));

  auto inst_norm =
      helper_->MakeNode("InstanceNormalization",
                        {reshape_in->output(0), dummy_scale, dummy_bias});
  AddAttribute(inst_norm, "epsilon", epsilon_);

  // Reshape back: [N, groups, -1] -> [N, C, H, W]
  auto origin_shape = helper_->MakeNode("Shape", {input_name})->output(0);
  auto reshape_out =
      helper_->MakeNode("Reshape", {inst_norm->output(0), origin_shape});

  std::string output = reshape_out->output(0);

  // Apply real scale/bias
  if (has_scale) {
    std::string us_scale = helper_->Unsqueeze(scale_name, {1, 2});
    output = helper_->MakeNode("Mul", {output, us_scale})->output(0);
  }
  if (has_bias) {
    std::string us_bias = helper_->Unsqueeze(bias_name, {1, 2});
    output = helper_->MakeNode("Add", {output, us_bias})->output(0);
  }

  // Squeeze back to 3D if PIR mode
  if (in_pir_mode) {
    output = helper_->Squeeze(output, {3});
  }

  helper_->MakeNode("Identity", {output}, {output_name});
}

}  // namespace paddle2onnx
