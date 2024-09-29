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

#include "paddle2onnx/mapper/tensor/full_int_array.h"

#include <iostream>
#include <string>
#include <vector>

namespace paddle2onnx {
REGISTER_PIR_MAPPER(full_int_array, FullIntArrayMapper)

void FullIntArrayMapper::Opset7() {
  auto output_info = GetOutput("Out");
  int64_t shape_dim = shape_values_.size();
  std::vector<int64_t> shape_ = {shape_dim};
  helper_->Assign(output_info[0].name, GetOnnxDtype(output_info[0].dtype),
                  shape_, shape_values_);
}

}