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

#include "paddle2onnx/mapper/tensor/masked_select.h"

namespace paddle2onnx {
REGISTER_PIR_MAPPER(masked_select, MaskedSelectMapper)

int32_t MaskedSelectMapper::GetMinOpsetVersion(bool verbose) { return 9; }

void MaskedSelectMapper::Opset9() {
  auto input_info = GetInput("x");
  auto mask_info = GetInput("mask");
  auto output_info = GetOutput("out");

  std::vector<int64_t> input_shape = input_info[0].shape;
  std::vector<int64_t> mask_shape = mask_info[0].shape;
  if (input_shape == mask_shape) {
    helper_->MakeNode("Compress",
                      {input_info[0].name, mask_info[0].name},
                      {output_info[0].name});
  } else {
    std::vector<int64_t> broadcasted_shape =
        helper_->GetBroadcastShape(input_shape, mask_shape);

    std::string broadcast_input = input_info[0].name;
    if (input_shape != broadcasted_shape) {
      broadcast_input = helper_->BroadcastTo(
          input_info[0].name, input_shape, broadcasted_shape);
    }

    std::string broadcast_mask = mask_info[0].name;
    if (mask_shape != broadcasted_shape) {
      broadcast_mask = helper_->BroadcastTo(
          mask_info[0].name, mask_shape, broadcasted_shape);
    }

    helper_->MakeNode(
        "Compress", {broadcast_input, broadcast_mask}, {output_info[0].name});
  }
}
}  // namespace paddle2onnx
