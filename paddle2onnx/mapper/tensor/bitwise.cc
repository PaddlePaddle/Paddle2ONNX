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

#include "paddle2onnx/mapper/tensor/bitwise.h"

namespace paddle2onnx {

REGISTER_MAPPER(bitwise_and, BitWiseMapper)
REGISTER_MAPPER(bitwise_not, BitWiseMapper)
REGISTER_MAPPER(bitwise_or, BitWiseMapper)
REGISTER_MAPPER(bitwise_xor, BitWiseMapper)

int32_t BitWiseMapper::GetMinOpsetVersion(bool verbose) {
  auto x_info = GetInput("X");
  if(x_info[0].dtype == P2ODataType::BOOL){
    Logger(verbose, 7) << RequireOpset(7) << std::endl;
    return 7;
  }
  Logger(verbose, 18) << RequireOpset(18) << std::endl;
  return 18;
}
void BitWiseMapper::Opset7() { 
  auto x_info = GetInput("X");
  auto out_info = GetOutput("Out");
  if (paddle_type_ == "bitwise_not"){
    helper_->MakeNode(onnx_elemwise_type_, {x_info[0].name}, {out_info[0].name});
  } else{
    auto y_info = GetInput("Y");
    helper_->MakeNode(onnx_elemwise_type_, {x_info[0].name, y_info[0].name}, {out_info[0].name});
  }
}

void BitWiseMapper::Opset18() {
  auto x_info = GetInput("X");
  auto out_info = GetOutput("Out");
  std::string node_name = x_info[0].dtype == P2ODataType::BOOL? onnx_elemwise_type_: onnx_bitwise_type_;
  if(paddle_type_ == "bitwise_not"){
    helper_->MakeNode(node_name, {x_info[0].name}, {out_info[0].name});
  } else{
    auto y_info = GetInput("Y");
    helper_->MakeNode(node_name, {x_info[0].name, y_info[0].name},{out_info[0].name});
  }
}
}// namespace paddle2onnx