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

#include "paddle2onnx/mapper/tensor/uniform.h"

namespace paddle2onnx {
REGISTER_PIR_MAPPER(uniform, UniformMapper)

int32_t UniformMapper::GetMinOpsetVersion(bool verbose) { return 7; }

void UniformMapper::Opset7() {
  auto output_info = GetOutput("out");
  auto shape_info = GetInput("shape");
  auto min_info = GetInput("min");
  auto max_info = GetInput("max");

  if (min_info[0].Rank() != 0 || max_info[0].Rank() != 0) {
    Error() << "[ERROR] min/max must be scalar tensors for op "
               "uniform "
            << std::endl;
  }
  std::vector<float> min_val{0.0f}, max_val{1.0f};
  bool is_min_const =
      helper_->TryGetTensorValue<float>(min_info[0].name, &min_val);
  bool is_max_const =
      helper_->TryGetTensorValue<float>(max_info[0].name, &max_val);

  std::vector<int64_t> shape_values;
  helper_->TryGetTensorValue<int64_t>(shape_info[0].name, &shape_values);

  auto onnx_dtype = GetOnnxDtype(dtype_);

  auto random_node =
      helper_->MakeNode("RandomUniform", {}, {output_info[0].name});

  AddAttribute(random_node, "shape", shape_values);
  AddAttribute(random_node, "low", min_val[0]);
  AddAttribute(random_node, "high", max_val[0]);
  AddAttribute(random_node, "dtype", static_cast<int64_t>(onnx_dtype));
  if (seed_ != 0) {
    AddAttribute(random_node, "seed", static_cast<float>(seed_));
  }
}
}  // namespace paddle2onnx
