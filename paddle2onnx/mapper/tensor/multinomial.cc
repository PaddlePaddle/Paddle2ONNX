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

namespace paddle2onnx {
REGISTER_MAPPER(multinomial, MultinomialMapper)

int32_t MultinomialMapper::GetMinOpset(bool verbose) {
  if (!replacement_ && num_samples_ > 1) {
    Error() << "replacement=False when num_samples > 1 is not supported for "
               "multinomial"
            << std::endl;
    return -1;
  }
  Logger(verbose, 7) << RequireOpset(7) << std::endl;
  return 7;
}

void MultinomialMapper::Opset7() {
  auto input_info = GetInput("X");
  auto output_info = GetOutput("Out");

  auto node = helper_->MakeNode("Multinomial", {input_info[0].name},
                                {output_info[0].name});
  AddAttribute(node, "num_samples", num_samples_);
  AddAttribute(node, "dtype", ONNX_NAMESPACE::TensorProto::INT64);
}

}  // namespace paddle2onnx
