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

#include "paddle2onnx/mapper/tensor/stack.h"

namespace paddle2onnx {
REGISTER_MAPPER(stack, StackMapper)
REGISTER_PIR_MAPPER(stack, StackMapper)

void StackMapper::Opset7() {
  auto x_info = GetInput("X");
  auto y_info = GetOutput("Y");

  int32_t out_dtype = 0;
  std::vector<std::string> aligned_inputs =
      helper_->DtypeAlignment(x_info, &out_dtype);

  // Find the maximum rank among all inputs based on TensorInfo
  int32_t max_rank = 0;
  for (size_t i = 0; i < x_info.size(); ++i) {
    int32_t rank = x_info[i].Rank();
    if (rank > max_rank) {
      max_rank = rank;
    }
  }

  // Special case: if all inputs are scalars [] or single-element tensors [1]
  // Check if all inputs have at most 1 element total
  bool all_single_element = true;
  for (size_t i = 0; i < x_info.size(); ++i) {
    if (x_info[i].Rank() == 0) {
      // Scalar, has 1 element - OK
      continue;
    } else if (x_info[i].Rank() == 1) {
      // Check if it's exactly [1] not [4] or other sizes
      if (x_info[i].shape.size() > 0 && x_info[i].shape[0] == 1) {
        // Single element [1] - OK
        continue;
      } else {
        // It's like [4] or [N] where N != 1 - NOT single element
        all_single_element = false;
        break;
      }
    } else {
      // Rank > 1, definitely not single element
      all_single_element = false;
      break;
    }
  }

  if (all_single_element && max_rank <= 1) {
    // All inputs are scalars or [1], normalize to scalars []
    for (size_t i = 0; i < aligned_inputs.size(); ++i) {
      aligned_inputs[i] =
          helper_->Reshape(aligned_inputs[i], std::vector<int64_t>{});
    }
    max_rank = 0;  // All are now scalars
  } else {
    // Normal case: make all inputs have the same rank by unsqueezing lower-rank
    // tensors
    for (size_t i = 0; i < aligned_inputs.size(); ++i) {
      int32_t rank_diff = max_rank - x_info[i].Rank();
      if (rank_diff > 0) {
        // Unsqueeze to match max_rank
        std::vector<int64_t> axes_to_add;
        for (int32_t j = 0; j < rank_diff; ++j) {
          axes_to_add.push_back(j);
        }
        aligned_inputs[i] = helper_->Unsqueeze(aligned_inputs[i], axes_to_add);
      }
    }
  }

  auto axis = axis_;
  if (axis < 0) {
    axis = axis + max_rank + 1;
  }

  // Now unsqueeze all inputs at the target axis for stacking
  for (size_t i = 0; i < aligned_inputs.size(); ++i) {
    aligned_inputs[i] =
        helper_->Unsqueeze(aligned_inputs[i], std::vector<int64_t>(1, axis));
  }
  auto out = helper_->Concat(aligned_inputs, axis);
  helper_->AutoCast(out, y_info[0].name, out_dtype, y_info[0].dtype);
}
}  // namespace paddle2onnx
