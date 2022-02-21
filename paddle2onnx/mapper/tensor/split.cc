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

#include "paddle2onnx/mapper/tensor/split.h"
#include <string>
#include <vector>

namespace paddle2onnx {
REGISTER_MAPPER(split, SplitMapper)

std::vector<int64_t> SplitMapper::GetAxes() {
  auto op = parser_->GetOpDesc(block_idx_, op_idx_);
  bool has_axis_tensor_input =
      parser_->OpHasInput(block_idx_, op_idx_, "AxisTensor");
  std::vector<int64_t> axis;
  if (has_axis_tensor_input) {
    std::vector<TensorInfo> axes_info =
        parser_->GetOpInput(block_idx_, op_idx_, "AxisTensor");
    std::vector<int64_t> index = parser_->GetBlockOpIdx(axes_info[0].name);
    Weight value;
    bool found_value = parser_->GetValueFromTensor(index[0], index[1], &value);
    if (found_value) {
      value.get(&axis);
    }
  } else {
    int64_t axis_val;
    parser_->GetOpAttr(op, "axis", &axis_val);
    axis.push_back(axis_val);
  }
  return axis;
}

int32_t SplitMapper::GetMinOpset(bool verbose) {
  auto op = parser_->GetOpDesc(block_idx_, op_idx_);
  auto axis = GetAxes();
  if (axis.size() == 0) {
    if (verbose) {
      std::cerr << "Currently does not support the axes parameter as input "
                   "tensor in op "
                << op.type() << "." << std::endl;
    }
    return -1;
  }
  return 7;
}

void SplitMapper::Opset7(OnnxHelper* helper) {
  auto op = parser_->GetOpDesc(block_idx_, op_idx_);
  std::vector<TensorInfo> input_info =
      parser_->GetOpInput(block_idx_, op_idx_, "X");
  std::vector<TensorInfo> output_info =
      parser_->GetOpOutput(block_idx_, op_idx_, "Out");
  std::vector<std::string> output_names;
  for (auto i : output_info) {
    output_names.push_back(i.name);
  }
  int64_t axis = GetAxes()[0];
  bool has_attr = parser_->OpHasAttr(op, "sections");
  if (has_attr) {
    std::vector<int64_t> sections;
    parser_->GetOpAttr(op, "sections", &sections);

    std::vector<int64_t> sections_index;
    for (auto i = 0; i < sections.size(); ++i) {
      if (sections[i] == -1) {
        sections_index.push_back(i);
      }
    }

    if (input_info[0].shape[axis] != -1 && sections_index.size() == 1) {
      int64_t sum_val = std::accumulate(sections.begin(), sections.end(), 0);
      sections[sections_index[0]] = input_info[0].shape[axis] - sum_val - 1;
    }

    helper->Split(input_info[0].name, output_names, sections, axis);
  } else {
    std::vector<int64_t> split_val;
    helper->Split(input_info[0].name, output_names, split_val, axis);
  }
}

}  // namespace paddle2onnx
