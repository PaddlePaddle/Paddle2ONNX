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

#include "paddle2onnx/mapper/tensor/isinf.h"

namespace paddle2onnx {
REGISTER_PIR_MAPPER(isinf, IsInfMapper)

int32_t IsInfMapper::GetMinOpsetVersion(bool verbose) { return 10; }

void IsInfMapper::Opset10() {
  auto input_info = GetInput("X");
  auto output_info = GetOutput("Out");
  if (input_info[0].dtype != P2ODataType::FP32 &&
      input_info[0].dtype != P2ODataType::FP64) {
    auto cast_type = P2ODataType::FP64;

    std::string cast_input =
        helper_->AutoCast(input_info[0].name, input_info[0].dtype, cast_type);
    helper_->MakeNode("IsInf", {cast_input}, {output_info[0].name});
  } else {
    helper_->MakeNode("IsInf", {input_info[0].name}, {output_info[0].name});
  }
}

void IsInfMapper::Opset20() {
  auto input_info = GetInput("X");
  auto output_info = GetOutput("Out");
  if (input_info[0].dtype != P2ODataType::FP16 &&
      input_info[0].dtype != P2ODataType::FP32 &&
      input_info[0].dtype != P2ODataType::FP64) {
    auto cast_type = P2ODataType::FP64;

    std::string cast_input =
        helper_->AutoCast(input_info[0].name, input_info[0].dtype, cast_type);
    helper_->MakeNode("IsInf", {cast_input}, {output_info[0].name});
  } else {
    helper_->MakeNode("IsInf", {input_info[0].name}, {output_info[0].name});
  }
}
}  // namespace paddle2onnx
