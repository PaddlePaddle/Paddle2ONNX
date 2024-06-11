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
#include <array>

namespace paddle2onnx {
REGISTER_MAPPER(bitwise_and, BitWiseAndMapper)

int32_t BitWiseAndMapper::GetMinOpset(bool verbose) {
  constexpr int op_version = 18;
  Logger(verbose, op_version) << RequireOpset(op_version) << std::endl;
  return op_version;
}
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
  bool is_cast = false;
  constexpr int cast_dtype = P2ODataType::UINT8;
  if (std::find(T.begin(), T.end(), x_info[0].dtype) == T.end()) {
    // Since x and y are of the same type, we only need to determine the type of x
    x_name = helper_->AutoCast(x_name, x_info[0].dtype, cast_dtype); // cast 
    y_name = helper_->AutoCast(y_name, y_info[0].dtype, cast_dtype); // cast
    is_cast = true;
  } 
  if(is_cast){
    auto bitwise_and_node = helper_->MakeNode("BitwiseAnd", {x_name, y_name});
    helper_->AutoCast(bitwise_and_node->output(0), out_info[0].name, cast_dtype, out_info[0].dtype);
    return;
  }
  helper_->MakeNode("BitwiseAnd", {x_name, y_name},{out_info[0].name});
}
}  // namespace paddle2onnx
