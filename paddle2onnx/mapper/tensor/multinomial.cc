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

#include "paddle2onnx/mapper/tensor/multinomial.h"
#include <iostream>
#include <string>
#include <vector>

namespace paddle2onnx {
REGISTER_PIR_MAPPER(multinomial, MultinomialMapper)

int32_t MultinomialMapper::GetMinOpsetVersion(bool verbose) {
  if (!IsConstantInput("num_samples")) {
    Error() << "num_samples is not a constant input." << std::endl;
    return -1;
  }
  return 7;
}

void MultinomialMapper::Opset7() {
  auto x_info = GetInput("x");
  auto out_info = GetOutput("out");
  double num_samples = 1;
  TryGetInputValue("num_samples", &num_samples);
  auto node =
      helper_->MakeNode("Multinomial", {x_info[0].name}, {out_info[0].name});
  AddAttribute(node, "dtype", GetOnnxDtype(out_info[0].dtype));
  AddAttribute(node, "sample_size", static_cast<int64_t>(num_samples));
  AddAttribute(node, "seed", static_cast<float>(0.0));
}

}  // namespace paddle2onnx
