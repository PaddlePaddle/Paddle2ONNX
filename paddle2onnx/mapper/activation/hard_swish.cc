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

#include "paddle2onnx/mapper/activation/hard_swish.h"

namespace paddle2onnx {
REGISTER_MAPPER(hard_swish, HardSwishMapper)

int32_t HardSwishMapper::GetMinOpsetVersion(bool verbose) {
    return 14;
}

void HardSwishMapper::Opset7() {
  auto input_info = GetInput("X");
  auto output_info = GetOutput("Out");

  std::string scale_node = helper_->Constant({}, GetOnnxDtype(input_info[0].dtype), scale_);
  std::string offset_node = helper_->Constant({}, GetOnnxDtype(input_info[0].dtype), offset_);
  auto add_node = helper_->MakeNode("Add", {input_info[0].name, offset_node});
  auto clip_node = helper_->Clip(add_node->output(0), 0.0, threshold_, input_info[0].dtype);
  auto mul_node = helper_->MakeNode("Mul", {input_info[0].name, clip_node});
  helper_->MakeNode("Div", {mul_node->output(0), scale_node}, {output_info[0].name});
}

inline bool IsAlmostEqual(float a, float b) {
  constexpr float epsilon = 1e-5f;
  return std::fabs(a - b) < epsilon;
}

void HardSwishMapper::Opset14() {
  if (!IsAlmostEqual(offset_, 3.0)) {
    P2OLogger() << "offset != 3.0, using Opset7()" << std::endl;
    return Opset7();
  }

  if (!IsAlmostEqual(scale_, 6.0)) {
    P2OLogger() << "scale_ != 6.0, using Opset7()" << std::endl;
    return Opset7();
  }

  if (!IsAlmostEqual(threshold_, 6.0)) {
    P2OLogger() << "offset != 3.0, using Opset7()" << std::endl;
    return Opset7();
  }
  auto input_info = GetInput("X");
  auto output_info = GetOutput("Out");
  helper_->MakeNode("HardSwish", {input_info[0].name}, {output_info[0].name});
}
}