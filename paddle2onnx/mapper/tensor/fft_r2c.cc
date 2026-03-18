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

#include "paddle2onnx/mapper/tensor/fft_r2c.h"

#include <cmath>
#include <string>
#include <vector>

namespace paddle2onnx {
REGISTER_PIR_MAPPER(fft_r2c, FftR2cMapper);

int32_t FftR2cMapper::GetMinOpsetVersion(bool verbose) { return 17; }

void FftR2cMapper::Opset17() {
  auto input_info = GetInput("x");
  auto output_info = GetOutput("out");
  output_info[0].dtype = P2ODataType::FP32;
  std::string one_str = helper_->Constant(GetOnnxDtype(P2ODataType::INT64),
                                          std::vector<int64_t>({-1}));
  std::string zero_str = helper_->Constant(GetOnnxDtype(P2ODataType::INT64),
                                           std::vector<int64_t>({0}));
  auto node1 = helper_->MakeNode("Unsqueeze", {input_info[0].name, one_str});
  auto node2 = helper_->MakeNode("Unsqueeze", {node1->output(0), zero_str});
  auto dft_node = helper_->MakeNode("DFT", {node2->output(0)});
  AddAttribute(dft_node, "onesided", int64_t(onesided_));
  AddAttribute(dft_node, "inverse", int64_t(0));
  AddAttribute(dft_node, "axis", int64_t(2));
  helper_->MakeNode(
      "Squeeze", {dft_node->output(0), zero_str}, {output_info[0].name});
}

}  // namespace paddle2onnx
