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
#include "paddle2onnx/mapper/nn/dropout.h"
#include <vector>

namespace paddle2onnx {

REGISTER_MAPPER(dropout, DropoutMapper)

int32_t DropoutMapper::GetMinOpset(bool verbose) {
  if (dropout_implementation_ != "downgrade_in_infer" &&
      dropout_implementation_ != "upscale_in_train") {
    if (verbose) {
      auto op = parser_->GetOpDesc(block_idx_, op_idx_);
      std::cerr << "dropout_implementation: " << dropout_implementation_
                << " is not support." << std::endl;
      std::cerr << " In op " << op.type()
                << " , dropout_implementation only support  downgrade_in_infer "
                   "and upscale_in_train. "
                << std::endl;
    }
    return -1;
  }
  return 7;
}

void DropoutMapper::Opset7(OnnxHelper* helper) {
  auto op = parser_->GetOpDesc(block_idx_, op_idx_);
  std::vector<TensorInfo> input_info =
      parser_->GetOpInput(block_idx_, op_idx_, "X");
  std::vector<TensorInfo> output_info =
      parser_->GetOpOutput(block_idx_, op_idx_, "Out");

  if (dropout_implementation_ == "upscale_in_train") {
    helper->MakeNode("Identity", {input_info[0].name}, {output_info[0].name});
  } else {
    std::vector<float> value = {1 - dropout_prob_};
    std::string scale_node =
        helper->Constant(GetOnnxDtype(input_info[0].dtype), value);
    helper->MakeNode("Mul", {input_info[0].name, scale_node},
                     {output_info[0].name});
  }
}

}  // namespace paddle2onnx
