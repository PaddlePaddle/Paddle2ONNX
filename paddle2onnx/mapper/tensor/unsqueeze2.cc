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
#include <string>
#include <vector>

namespace paddle2onnx {
REGISTER_MAPPER(unsqueeze2, UnSqueeze2Mapper)

int32_t UnSqueeze2Mapper::GetMinOpset(bool verbose) {
  auto op = parser_->GetOpDesc(block_idx_, op_idx_);
  bool has_attr = parser_->OpHasAttr(op, "axes");
  if (has_attr) return 7;
  bool has_axis_tensor_info =
      parser_->OpHasInput(block_idx_, op_idx_, "AxesTensor");
  if (!has_axis_tensor_info) {
    if (verbose) {
      std::cerr << " Can not find Axes as input or attribute in op "
                << op.type() << "." << std::endl;
    }
    return -1;
  }
  std::vector<TensorInfo> axes_info =
      parser_->GetOpInput(block_idx_, op_idx_, "AxesTensor");
  std::vector<int64_t> index = parser_->GetBlockOpIdx(axes_info[0].name);
  Weight value;
  bool found_value = parser_->GetValueFromTensor(index[0], index[1], &value);

  if (!found_value) {
    if (verbose) {
      std::cerr << "Currently does not support the axes parameter as input "
                   "tensor in op "
                << op.type() << "." << std::endl;
    }
    return -1;
  } else {
    std::vector<int64_t> axes = ComputeAxes();
    bool sorted = std::is_sorted(axes.begin(), axes.end());
    if (!sorted) {
      if (verbose) {
        std::cerr << " axes must be arranged in the following order in op "
                  << op.type() << "." << std::endl;
      }
      return -1;
    }
  }
  return 7;
}

std::vector<int64_t> UnSqueeze2Mapper::ComputeAxes() {
  auto op = parser_->GetOpDesc(block_idx_, op_idx_);

  bool has_attr = parser_->OpHasAttr(op, "axes");
  std::vector<int64_t> axes;
  if (has_attr) {
    parser_->GetOpAttr(op, "axes", &axes);
  } else {
    std::vector<TensorInfo> axes_info =
        parser_->GetOpInput(block_idx_, op_idx_, "AxesTensor");
    std::vector<int64_t> index = parser_->GetBlockOpIdx(axes_info[0].name);
    Weight value;
    parser_->GetValueFromTensor(index[0], index[1], &value);
    value.get(axes);
  }
  std::vector<TensorInfo> input_info =
      parser_->GetOpInput(block_idx_, op_idx_, "X");
  for (auto i = 0; i < axes.size(); ++i) {
    if (axes[i] < 0) {
      axes[i] = axes[i] + input_info[0].Rank() + i + 1;
    }
  }
  return axes;
}

void UnSqueeze2Mapper::Opset7(OnnxHelper* helper) {
  auto op = parser_->GetOpDesc(block_idx_, op_idx_);
  std::vector<TensorInfo> input_info =
      parser_->GetOpInput(block_idx_, op_idx_, "X");
  std::vector<TensorInfo> output_info =
      parser_->GetOpOutput(block_idx_, op_idx_, "Out");

  auto axes = ComputeAxes();
  if (export_opset_version_ < 13) {
    auto node = helper->MakeNode("Unsqueeze", {input_info[0].name},
                                 {output_info[0].name});
    AddAttribute(node, "axes", axes);
  } else {
    std::string axes_node =
        helper->Constant(GetOnnxDtype(P2ODataType::INT64), axes);
    auto node = helper->MakeNode("Unsqueeze", {input_info[0].name, axes_node},
                                 {output_info[0].name});
  }
}

}  // namespace paddle2onnx
