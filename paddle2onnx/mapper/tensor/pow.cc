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

#include "paddle2onnx/mapper/tensor/pow.h"

#include <unordered_set>

namespace paddle2onnx {
REGISTER_MAPPER(pow, PowMapper)

void PowMapper::Opset7() {
  auto input_info = GetInput("X");
  auto output_info = GetOutput("Out");

  auto factor_node =
      helper_->Constant({}, ONNX_NAMESPACE::TensorProto::FLOAT, factor_);
  //  std::vector<float>(1, factor_));
  if (input_info[0].dtype != P2ODataType::FP32) {
    std::string x_cast_name = helper_->AutoCast(
        {input_info[0].name}, input_info[0].dtype, P2ODataType::FP32);
    auto node = helper_->MakeNode("Pow", {x_cast_name, factor_node});
    std::string squeeze = node->output(0);
    if (!input_info[0].Rank()) {
      squeeze = helper_->Squeeze(node->output(0), {0});
    }
    helper_->AutoCast(squeeze, {output_info[0].name}, P2ODataType::FP32,
                      input_info[0].dtype);
  } else {
    std::string input_name = input_info[0].name;
    if (!input_info[0].Rank()) {
      input_name = helper_->Squeeze(input_info[0].name, {0});
    }
    helper_->MakeNode("Pow", {input_name, factor_node}, {output_info[0].name});
  }
}

}  // namespace paddle2onnx
