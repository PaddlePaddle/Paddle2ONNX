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

#include "paddle2onnx/mapper/tensor/full.h"
#include <iostream>
#include <string>
#include <type_traits>
#include <vector>

namespace paddle2onnx {
REGISTER_PIR_MAPPER(full, FullMapper)

void FullMapper::Opset7() {
  auto output_info = GetOutput("Out");
  std::visit(
      [&](auto&& arg) {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<T, double>) {
          helper_->Constant(output_info[0].name,
                            shape_,
                            GetOnnxDtype(output_info[0].dtype),
                            std::get<double>(value_));

        } else if constexpr (std::is_same_v<T, float>) {
          helper_->Constant(output_info[0].name,
                            shape_,
                            GetOnnxDtype(output_info[0].dtype),
                            std::get<float>(value_));
        } else if constexpr (std::is_same_v<T, int64_t>) {
          helper_->Constant(output_info[0].name,
                            shape_,
                            GetOnnxDtype(output_info[0].dtype),
                            std::get<int64_t>(value_));
        } else if constexpr (std::is_same_v<T, int32_t>) {
          helper_->Constant(output_info[0].name,
                            shape_,
                            GetOnnxDtype(output_info[0].dtype),
                            std::get<int32_t>(value_));
        }
      },
      value_);
}
}  // namespace paddle2onnx
