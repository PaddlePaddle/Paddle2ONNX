// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle2onnx/mapper/activation/hardtanh.h"

namespace paddle2onnx {
REGISTER_PIR_MAPPER(hardtanh, HardtanhMapper)

int32_t HardtanhMapper::GetMinOpsetVersion(bool verbose) {
  Logger(verbose, 7) << RequireOpset(7) << std::endl;
  return 7;
}

void HardtanhMapper::Opset7() {
  auto input_info = GetInput("x");
  auto output_info = GetOutput("out");

  helper_->Clip(
      input_info[0].name, output_info[0].name, min_, max_, input_info[0].dtype);
}
}  // namespace paddle2onnx
