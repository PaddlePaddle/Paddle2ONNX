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

#include "paddle2onnx/mapper/tensor/unbind.h"

namespace paddle2onnx {
REGISTER_MAPPER(unbind, UnbindMapper)
REGISTER_PIR_MAPPER(unbind, UnbindMapper)

void UnbindMapper::Opset7() {
  auto input_info = GetInput("X");
  auto output_info = GetOutput("Out");

  std::vector<std::string> output_names(output_info.size());
  for (size_t i = 0; i < output_info.size(); ++i) {
    output_names[i] = output_info[i].name;
  }

  int64_t split_axis = axis_;
  if (split_axis < 0) {
    split_axis += input_info[0].Rank();
  }

  std::vector<int64_t> split_sizes =
      std::vector<int64_t>(input_info[0].shape[split_axis], 1);
  auto split_output_names =
      helper_->Split(input_info[0].name, split_sizes, split_axis);

  for (size_t i = 0; i < output_info.size(); ++i) {
    std::vector<int64_t> axes{split_axis};
    helper_->Squeeze(split_output_names[i], output_names[i], axes);
  }
}
}  // namespace paddle2onnx
