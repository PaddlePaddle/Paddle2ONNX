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

#include "paddle2onnx/mapper/tensor/greater_than.h"

namespace paddle2onnx {
REGISTER_MAPPER(greater_than, GreaterThanMapper)
int32_t GreaterThanMapper::GetMinOpsetVersion(bool verbose) {
  // NHWC is not supported
  auto x_info = GetInput("X");
  auto y_info = GetInput("Y");

  if (x_info[0].dtype == P2ODataType::BOOL || y_info[0].dtype == P2ODataType::BOOL) {
      Logger(verbose, 9) << "While the type of input is (bool), " << RequireOpset(9) << std::endl;
      return 9;
  }
  return 7;
}
void GreaterThanMapper::Opset7() {
  auto x_info = GetInput("X");
  auto y_info = GetInput("Y");
  auto out_info = GetOutput("Out");


  int out_dtype = 0;
  std::vector<std::string> aligned_inputs = helper_->DtypeAlignment({x_info[0], y_info[0]}, &out_dtype);

  if (out_dtype == P2ODataType::BOOL){ 
    std::string new_x_name = helper_->AutoCast(x_info[0].name, x_info[0].dtype, P2ODataType::INT32);
    std::string new_y_name = helper_->AutoCast(y_info[0].name, y_info[0].dtype, P2ODataType::INT32);
    helper_->MakeNode("Greater", {new_x_name, new_y_name}, {out_info[0].name});
    return ;
  }
  if (out_dtype != P2ODataType::FP32 && out_dtype != P2ODataType::FP64 &&
      helper_->GetOpsetVersion() < 11) {
    aligned_inputs[0] =
        helper_->AutoCast(aligned_inputs[0], out_dtype, P2ODataType::FP32);
    aligned_inputs[1] =
        helper_->AutoCast(aligned_inputs[1], out_dtype, P2ODataType::FP32);
  }


  helper_->MakeNode("Greater", aligned_inputs, {out_info[0].name});
}

}  // namespace paddle2onnx