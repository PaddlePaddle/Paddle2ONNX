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

#include "paddle2onnx/mapper/tensor/assign_value.h"
#include <iostream>
#include <string>
#include <vector>

namespace paddle2onnx {
REGISTER_MAPPER(assign_value, AssignValueMapper)

int32_t AssignValueMapper::GetMinOpset(bool verbose) {
  auto op = parser_->GetOpDesc(block_idx_, op_idx_);
  bool has_input = parser_->OpHasInput(block_idx_, op_idx_, "X");
  bool found_value = parser_->GetValueFromTensor(block_idx_, op_idx_);
  if (!has_input && !found_value) {
    if (verbose) {
      std::cerr << "Can not find input and value attribute in op " << op.type()
                << "." << std::endl;
    }
    return -1;
  }
  return 7;
}

void AssignValueMapper::Opset7(OnnxHelper* helper) {
  auto op = parser_->GetOpDesc(block_idx_, op_idx_);
  if (parser_->OpHasInput(block_idx_, op_idx_, "X")) {
    std::vector<TensorInfo> input_info =
        parser_->GetOpInput(block_idx_, op_idx_, "X");
    std::vector<TensorInfo> output_info =
        parser_->GetOpOutput(block_idx_, op_idx_, "Out");
    helper->MakeNode("Identity", {input_info[0].name}, {output_info[0].name});
  } else {
    std::vector<TensorInfo> output_info =
        parser_->GetOpOutput(block_idx_, op_idx_, "Out");

    Weight param;
    parser_->GetValueFromTensor(block_idx_, op_idx_, &param);
    auto node = helper->MakeConstant(param);
    helper->AutoCast(node->output(0), output_info[0].name, param.dtype,
                     output_info[0].dtype);
  }
}

}  // namespace paddle2onnx
