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

#include "paddle2onnx/mapper/nn/batch_norm.h"

#include <string>
#include <vector>

namespace paddle2onnx {
REGISTER_MAPPER(batch_norm, BatchNormMapper)
REGISTER_PIR_MAPPER(batch_norm, BatchNormMapper)

int32_t BatchNormMapper::GetMinOpsetVersion(bool verbose) {
  if (!use_global_stats_) {
    Logger(verbose, 14) << RequireOpset(14) << std::endl;
    return 14;
  }
  return 7;
}

void BatchNormMapper::Opset7() {
  auto input_info = GetInput("X");
  auto mean_info = GetInput("Mean");
  auto variance_info = GetInput("Variance");
  auto output_info = GetOutput("Y");

  std::string scale_name, bias_name;
  int64_t numel = 1;
  for (auto s : mean_info[0].shape) {
    numel *= s;
  }
  if (HasInput("Scale")) {
    scale_name = GetInput("Scale")[0].name;
  } else {
    std::vector<int64_t> values(numel, 1);
    scale_name = helper_->Constant(
        mean_info[0].shape, GetOnnxDtype(mean_info[0].dtype), values);
  }

  if (HasInput("Bias")) {
    bias_name = GetInput("Bias")[0].name;
  } else {
    std::vector<int64_t> values(numel, 0);
    bias_name = helper_->Constant(
        mean_info[0].shape, GetOnnxDtype(mean_info[0].dtype), values);
  }

  auto node = helper_->MakeNode("BatchNormalization",
                                {input_info[0].name,
                                 scale_name,
                                 bias_name,
                                 mean_info[0].name,
                                 variance_info[0].name},
                                {output_info[0].name});
  if (helper_->GetOpsetVersion() < 9) {
    int64_t spatial = 1;
    AddAttribute(node, "spatial", spatial);
  }

  AddAttribute(node, "epsilon", epsilon_);
  AddAttribute(node, "momentum", momentum_);
}

void BatchNormMapper::Opset14() {
  auto input_info = GetInput("X");
  auto mean_info = GetInput("Mean");
  auto variance_info = GetInput("Variance");
  auto output_info = GetOutput("Y");
  auto mean_out_info = GetOutput("MeanOut");
  auto variance_out_info = GetOutput("VarianceOut");

  std::string scale_name, bias_name;
  int64_t numel = 1;
  for (auto s : mean_info[0].shape) {
    numel *= s;
  }
  if (HasInput("Scale")) {
    scale_name = GetInput("Scale")[0].name;
  } else {
    std::vector<int64_t> values(numel, 1);
    scale_name = helper_->Constant(
        mean_info[0].shape, GetOnnxDtype(mean_info[0].dtype), values);
  }

  if (HasInput("Bias")) {
    bias_name = GetInput("Bias")[0].name;
  } else {
    std::vector<int64_t> values(numel, 0);
    bias_name = helper_->Constant(
        mean_info[0].shape, GetOnnxDtype(mean_info[0].dtype), values);
  }

  std::vector<std::string> output_names;
  output_names.push_back(output_info[0].name);
  if (!use_global_stats_) {
    output_names.push_back(mean_out_info[0].name);
    output_names.push_back(variance_out_info[0].name);
  }
  auto node = helper_->MakeNode("BatchNormalization",
                                {input_info[0].name,
                                 scale_name,
                                 bias_name,
                                 mean_info[0].name,
                                 variance_info[0].name},
                                output_names);
  if (helper_->GetOpsetVersion() < 9) {
    int64_t spatial = 1;
    AddAttribute(node, "spatial", spatial);
  }

  AddAttribute(node, "epsilon", epsilon_);
  AddAttribute(node, "momentum", momentum_);
  AddAttribute(node, "training_mode", static_cast<int64_t>(!use_global_stats_));
}

}  // namespace paddle2onnx
