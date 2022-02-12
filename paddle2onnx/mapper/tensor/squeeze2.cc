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

#include "paddle2onnx/mapper/tensor/squeeze2.h"
#include <string>
#include <vector>

namespace paddle2onnx {
REGISTER_MAPPER(squeeze2, Squeeze2Mapper)

std::vector<int64_t> Squeeze2Mapper::comput_axes() {
  auto op = parser_->GetOpDesc(block_idx_, op_idx_);
  std::vector<TensorInfo> input_info =
      parser_->GetOpInput(block_idx_, op_idx_, "X");
  std::vector<int64_t> axes;
  if (parser_->OpHasAttr(op, "axes")) {
    parser_->GetOpAttr(op, "axes", &axes);
    for (auto& i : axes) {
      if (i < 0) i = i + input_info[0].Rank();
    }
  }
  return axes;
}

void Squeeze2Mapper::Opset7(OnnxHelper* helper) {
  auto op = parser_->GetOpDesc(block_idx_, op_idx_);
  std::vector<TensorInfo> input_info =
      parser_->GetOpInput(block_idx_, op_idx_, "X");
  std::vector<TensorInfo> output_info =
      parser_->GetOpOutput(block_idx_, op_idx_, "Out");

  std::vector<int64_t> ret;
  ret.reserve(input_info[0].shape.size());
  for (auto i : input_info[0].shape) {
    if (i > 1) ret.push_back(i);
  }
  if (ret.size() == input_info[0].shape.size()) {
    helper->MakeNode("Identity", {input_info[0].name}, {output_info[0].name});
  } else {
    auto axes = comput_axes();
    if (axes.size() > 0) {
      std::sort(axes.begin(), axes.end());
      helper->Squeeze(input_info[0].name, output_info[0].name, axes);
    } else {
      helper->Squeeze(input_info[0].name, output_info[0].name, {});
    }
  }
}

}  // namespace paddle2onnx
