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

#include "paddle2onnx/mapper/tensor/bitwise_and.h"

namespace paddle2onnx {
REGISTER_MAPPER(bitwise_and, BitWiseAndMapper)

int32_t BitWiseAndMapper::GetMinOpset(bool verbose) {
  constexpr int op_version = 18;
  Logger(verbose, op_version) << RequireOpset(op_version) << std::endl;
  return op_version;
}

/**
BOOL,
  INT16,
  INT32,
  INT64,
  FP16,
  FP32,
  FP64,
  X7,
  X8,
  X9,
  X10,
  X11,
  X12,
  X13,
  X14,
  X15,
  X16,
  X17,
  X18,
  X19,
  UINT8,
  INT8,
*/
void BitWiseAndMapper::Opset18() {
  auto x_info = GetInput("X");
  auto y_info = GetInput("Y");
  auto out_info = GetOutput("Out");
  constexpr std::array<P2ODataType, 5> T = {
    P2ODataType::INT16,
    P2ODataType::INT32,
    P2ODataType::INT64,
    P2ODataType::INT8,
    P2ODataType::UINT8,
    // Except BOOL. Cast to UINT8.
  };
  std::string x_name = x_info[0].name;
  std::string y_name = y_info[0].name;
  if (std::find(T.begin(), T.end(), x_info[0].dtype) == T.end()) {
    x_name = helper_->AutoCast(x_name, x_info[0].dtype, P2ODataType::UINT8); // cast 
  }
  if (std::find(T.begin(), T.end(), y_info[0].dtype) == T.end()) {
    y_name = helper_->AutoCast(y_name, y_info[0].dtype, P2ODataType::UINT8); // cast 
  }
  auto output = helper_->MakeNode("BitwiseAnd", {x_name, y_name}, {out_info[0].name});
}
}  // namespace paddle2onnx
