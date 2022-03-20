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

#include "paddle2onnx/mapper/tensor/unsqueeze2.h"

namespace paddle2onnx {
REGISTER_MAPPER(unsqueeze2, Unsqueeze2Mapper)

int32_t Unsqueeze2Mapper::GetMinOpset(bool verbose) {
  auto op = parser_->GetOpDesc(block_idx_, op_idx_);
  if (axes_.size() == 0) {
    if (parser_->OpHasInput(block_idx_, op_idx_, "AxesTensorList")) {
      if (verbose) {
        std::cerr << "[Paddle2ONNX] AxesTensorList as input is not support for "
                     "op unsqueeze2 with opset < 13."
                  << std::endl;
      }
      return 13;
    } else if (parser_->OpHasInput(block_idx_, op_idx_, "AxesTensor")) {
      auto info = parser_->GetOpInput(block_idx_, op_idx_, "AxesTensor");
      if (!parser_->IsConstantTensor(block_idx_, info[0].name)) {
        if (verbose) {
          std::cerr << "[Paddle2ONNX] AxesTensor as a nonconstant tensor input "
                       "is not support for op unsqueeze2 with opset < 13."
                    << std::endl;
        }
        return 13;
      }
      if (!parser_->IsConstantTensor(block_idx_, info[0].name)) {
        return 13;
      } else {
        return 7;
      }
    }
    if (verbose) {
      std::cerr << "[Paddle2ONNX] Cannot found axes for op unsqueeze2."
                << std::endl;
    }
    return -1;
  }
  return 7;
}

void Unsqueeze2Mapper::Opset7(OnnxHelper* helper) {
  auto op = parser_->GetOpDesc(block_idx_, op_idx_);
  std::vector<TensorInfo> input_info =
      parser_->GetOpInput(block_idx_, op_idx_, "X");
  std::vector<TensorInfo> output_info =
      parser_->GetOpOutput(block_idx_, op_idx_, "Out");

  std::vector<int64_t> axes;
  if (axes_.empty()) {
    auto info = parser_->GetOpInput(block_idx_, op_idx_, "AxesTensor");
    Assert(parser_->TryGetTensorValue(block_idx_, info[0].name, &axes),
           "While unsqueeze2 has input AxesTensor, it cannot be exported by "
           "Paddle2ONNX");
  } else {
    axes.assign(axes_.begin(), axes_.end());
  }
  for (size_t i = 0; i < axes.size(); ++i) {
    if (axes[i] < 0) {
      axes[i] = axes[i] + input_info[0].Rank() + i + 1;
    }
  }
  helper->Unsqueeze(input_info[0].name, output_info[0].name, axes);
}

void Unsqueeze2Mapper::Opset13(OnnxHelper* helper) {
  auto op = parser_->GetOpDesc(block_idx_, op_idx_);
  std::vector<TensorInfo> input_info =
      parser_->GetOpInput(block_idx_, op_idx_, "X");
  std::vector<TensorInfo> output_info =
      parser_->GetOpOutput(block_idx_, op_idx_, "Out");

  std::vector<int64_t> axes;
  if (axes_.empty()) {
    auto info = parser_->GetOpInput(block_idx_, op_idx_, "AxesTensor");
    parser_->TryGetTensorValue(block_idx_, info[0].name, &axes);
  } else {
    axes.assign(axes_.begin(), axes_.end());
  }
  for (size_t i = 0; i < axes.size(); ++i) {
    if (axes[i] < 0) {
      axes[i] = axes[i] + input_info[0].Rank() + i + 1;
    }
  }

  if (axes.size() > 0) {
    helper->Unsqueeze(input_info[0].name, output_info[0].name, axes);
  } else {
    std::string axes_node = "";
    if (parser_->OpHasInput(block_idx_, op_idx_, "AxesTensorList")) {
      auto info = parser_->GetOpInput(block_idx_, op_idx_, "AxesTensorList");
      axes_node = helper->ConcatIndices(info);
    } else {
      auto info = parser_->GetOpInput(block_idx_, op_idx_, "AxesTensor");
      axes_node =
          helper->AutoCast(info[0].name, info[0].dtype, P2ODataType::INT64);
    }
    helper->MakeNode("Unsqueeze", {input_info[0].name, axes_node},
                     {output_info[0].name});
  }
}

}  // namespace paddle2onnx
