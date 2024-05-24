// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle2onnx/mapper/nn/deform_conv2d.h"

#include <string>
#include <vector>

namespace paddle2onnx {
REGISTER_MAPPER(deformable_conv, DeformConv2dMapper)

int32_t DeformConv2dMapper::GetMinOpset(bool verbose) {
  return 19;
}

void DeformConv2dMapper::Opset19() {
  auto input_info = GetInput("Input");
  auto kernel_info = GetInput("Filter");
  auto offset_info = GetInput("Offset");
  auto mask_info = GetInput("Mask");
  auto output_info = GetOutput("Output");
  std::string bias_name = helper_->Constant({kernel_info[0].shape[0]}, GetOnnxDtype(input_info[0].dtype), static_cast<float>(0.0));
  auto node = helper_->MakeNode(
    "DeformConv",
    {input_info[0].name, kernel_info[0].name, offset_info[0].name, bias_name, mask_info[0].name},
    {output_info[0].name});

  AddAttribute(node, "dilations", dilations_);
  AddAttribute(node, "group", groups_);
  // std::vector<int64_t> kernel_shape = {
  //   kernel_info[0].shape[2],
  //   kernel_info[0].shape[3]
  // };
  // AddAttribute(node, "kernel_shape", kernel_shape);
  std::vector<int64_t> paddings;
  if (paddings_.size() == 2) {
    paddings.insert(paddings.begin(), paddings_.begin(), paddings_.end());
    paddings.insert(paddings.begin(), paddings_.begin(), paddings_.end());
  } else {
    paddings.assign(paddings_.begin(), paddings_.end());
    paddings[1] = paddings_[2];
    paddings[2] = paddings_[1];
  }
  AddAttribute(node, "pads", paddings);
  AddAttribute(node, "strides", strides_);
}
} // namespace paddle2onnx
