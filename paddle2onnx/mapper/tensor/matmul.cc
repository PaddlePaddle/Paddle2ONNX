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

#include "paddle2onnx/mapper/tensor/matmul.h"
#include <iostream>
#include <string>
#include <vector>

namespace paddle2onnx {
REGISTER_MAPPER(matmul, MatmulMapper)

std::string MatmulMapper::GetTrans(std::vector<TensorInfo>& input_info,
                                   OnnxHelper* helper) {
  std::string castd_name = input_info[0].name;
  if (input_info[0].dtype == P2ODataType::FP64) {
    castd_name = helper->AutoCast(input_info[0].name, input_info[0].dtype,
                                  P2ODataType::FP32);
  }
  std::vector<int64_t> perm = Arange(0, input_info[0].Rank());
  std::swap(perm[perm.size() - 1], perm[perm.size() - 2]);
  auto transpose_node = helper->MakeNode("Transpose", {castd_name});
  AddAttribute(transpose_node, "perm", perm);
  return transpose_node->output(0);
}

void MatmulMapper::Opset7(OnnxHelper* helper) {
  auto op = parser_->GetOpDesc(block_idx_, op_idx_);
  std::vector<TensorInfo> input_x_info =
      parser_->GetOpInput(block_idx_, op_idx_, "X");
  std::vector<TensorInfo> input_y_info =
      parser_->GetOpInput(block_idx_, op_idx_, "Y");
  std::vector<TensorInfo> output_info =
      parser_->GetOpOutput(block_idx_, op_idx_, "Out");
  std::string input_x = input_x_info[0].name;
  if (transpose_X_) {
    input_x = GetTrans(input_x_info, helper);
  }
  std::string input_y = input_y_info[0].name;
  if (transpose_Y_) {
    input_y = GetTrans(input_y_info, helper);
  }
  if (abs(alpha_ - 1.0) < 1e-6) {
    auto node = helper->MakeNode("MatMul", {input_x, input_y});
    helper->AutoCast(node->output(0), output_info[0].name, P2ODataType::FP32,
                     input_y_info[0].dtype);
  } else {
    auto mutmul_node = helper->MakeNode("MatMul", {input_x, input_y});
    std::string scale_node =
        helper->Constant({1}, GetOnnxDtype(input_x_info[0].dtype), alpha_);
    auto mul_node =
        helper->MakeNode("Mul", {mutmul_node->output(0), scale_node});
    helper->AutoCast(mul_node->output(0), output_info[0].name,
                     P2ODataType::FP32, input_y_info[0].dtype);
  }
}

}  // namespace paddle2onnx
