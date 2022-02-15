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

#include "paddle2onnx/mapper/nn/batch_norm.h"
#include <string>
#include <vector>

namespace paddle2onnx {
REGISTER_MAPPER(batch_norm, BatchNormMapper)

void BatchNormMapper::Opset7(OnnxHelper* helper) {
  std::vector<TensorInfo> input_info =
      parser_->GetOpInput(block_idx_, op_idx_, "X");
  std::vector<TensorInfo> scale_info =
      parser_->GetOpInput(block_idx_, op_idx_, "Scale");
  std::vector<TensorInfo> bias_info =
      parser_->GetOpInput(block_idx_, op_idx_, "Bias");
  std::vector<TensorInfo> mean_info =
      parser_->GetOpInput(block_idx_, op_idx_, "Mean");
  std::vector<TensorInfo> variance_info =
      parser_->GetOpInput(block_idx_, op_idx_, "Variance");
  std::vector<TensorInfo> output_info =
      parser_->GetOpOutput(block_idx_, op_idx_, "Y");

  auto node = helper->MakeNode(
      "BatchNormalization",
      {input_info[0].name, scale_info[0].name, bias_info[0].name,
       mean_info[0].name, variance_info[0].name},
      {output_info[0].name});
  if (export_opset_version_ < 9) {
    int64_t spatial = 1;
    AddAttribute(node, "spatial", spatial);
  }

  AddAttribute(node, "epsilon", epsilon_);
  AddAttribute(node, "momentum", momentum_);
}

}  // namespace paddle2onnx
