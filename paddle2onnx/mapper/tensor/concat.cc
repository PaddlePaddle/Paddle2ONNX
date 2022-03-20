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

#include "paddle2onnx/mapper/tensor/concat.h"
#include <string>
#include <vector>

namespace paddle2onnx {
REGISTER_MAPPER(concat, ConcatMapper)

int32_t ConcatMapper::GetMinOpset(bool verbose) {
  if (parser_->OpHasInput(block_idx_, op_idx_, "AxisTensor")) {
    auto info = parser_->GetOpInput(block_idx_, op_idx_, "AxisTensor");
    if (!parser_->IsConstantTensor(block_idx_, info[0].name)) {
      std::cerr << "[Paddle2ONNX] While AxisTensor as input exists, it's not "
                   "supported unless it's a constant tensor for op concat."
                << std::endl;
      return -1;
    }
  }
  return 7;
}

void ConcatMapper::Opset7(OnnxHelper* helper) {
  auto op = parser_->GetOpDesc(block_idx_, op_idx_);
  std::vector<TensorInfo> input_info =
      parser_->GetOpInput(block_idx_, op_idx_, "X");
  std::vector<TensorInfo> output_info =
      parser_->GetOpOutput(block_idx_, op_idx_, "Out");

  int32_t casted_dtype;
  std::vector<std::string> casted_names =
      helper->DtypeAlignment(input_info, &casted_dtype);
  bool has_axis_tensor_input =
      parser_->OpHasInput(block_idx_, op_idx_, "AxisTensor");

  int64_t axis = axis_;
  if (parser_->OpHasInput(block_idx_, op_idx_, "AxisTensor")) {
    auto info = parser_->GetOpInput(block_idx_, op_idx_, "AxisTensor");
    std::vector<int64_t> value;
    Assert(parser_->TryGetTensorValue(block_idx_, info[0].name, &value),
           "While concat has input AxisTensor, and it's not a constant tensor, "
           "the model cannot be converted.");
    axis = value[0];
  }
  if (axis < 0) {
    axis = axis + input_info[0].Rank();
  }
  auto node = helper->MakeNode("Concat", casted_names);
  AddAttribute(node, "axis", axis);
  helper->AutoCast(node->output(0), output_info[0].name, casted_dtype,
                   output_info[0].dtype);
}

}  // namespace paddle2onnx
