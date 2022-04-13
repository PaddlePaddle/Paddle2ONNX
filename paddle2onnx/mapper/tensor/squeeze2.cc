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

#include "paddle2onnx/mapper/tensor/squeeze2.h"

namespace paddle2onnx {
REGISTER_MAPPER(squeeze2, Squeeze2Mapper)

void Squeeze2Mapper::Opset7() {
  auto input_info = GetInput("X");
  auto output_info = GetOutput("Out");

  std::vector<int64_t> ret;
  ret.reserve(input_info[0].shape.size());
  for (auto i : input_info[0].shape) {
    if (i > 1) ret.push_back(i);
  }
  if (ret.size() == input_info[0].Rank()) {
    helper_->MakeNode("Identity", {input_info[0].name}, {output_info[0].name});
  } else {
    std::vector<int64_t> axes(axes_.begin(), axes_.end());
    for (size_t i = 0; i < axes.size(); ++i) {
      if (axes[i] < 0) {
        axes[i] += input_info[0].Rank();
      }
    }
    if (axes.size() > 0) {
      std::sort(axes.begin(), axes.end());
      helper_->Squeeze(input_info[0].name, output_info[0].name, axes);
    } else {
      helper_->Squeeze(input_info[0].name, output_info[0].name, {});
    }
  }
}

}  // namespace paddle2onnx
