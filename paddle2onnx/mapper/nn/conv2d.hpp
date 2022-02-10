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

#pragma once
#include <string>
#include <vector>
#include "paddle2onnx/mapper/mapper.hpp"

namespace paddle2onnx {

class Conv2dMapper : public Mapper {
 public:
  Conv2dMapper(const PaddleParser& p, int64_t block_id, int64_t op_id)
      : Mapper(p, block_id, op_id) {
    auto op = parser_->GetOpDesc(block_idx_, op_idx_);
    parser_->GetOpAttr(op, "groups", &groups_);
    parser_->GetOpAttr(op, "dilations", &dilations_);
    parser_->GetOpAttr(op, "strides", &strides_);
    parser_->GetOpAttr(op, "paddings", &paddings_);
    parser_->GetOpAttr(op, "padding_algorithm", &padding_algorithm_);
    parser_->GetOpAttr(op, "data_format", &data_format_);
    if (paddings_.size() == 2) {
      paddings_.push_back(paddings_[0]);
      paddings_.push_back(paddings_[1]);
    } else if (paddings_.size() == 4) {
      int32_t tmp = paddings_[1];
      paddings_[1] = paddings_[2];
      paddings_[2] = tmp;
    }
  }

  int32_t GetMinOpset(bool verbose = false) {
    // NHWC is not supported
    if (data_format_ == "NHWC") {
      if (verbose) {
        std::cerr << "[ERROR] Cannot support NHWC format for operator conv2d."
                  << std::endl;
      }
      return -1;
    }
    // strides should be less or equal than kernel size
    auto kernel_info = parser_->GetOpInput(block_idx_, op_idx_, "Filter");
    if (kernel_info[0].shape[2] < strides_[0] ||
        kernel_info[0].shape[3] < strides_[1]) {
      if (verbose) {
        std::cerr
            << "[ERROR] Cannot handle the situation that kernel_size < strides"
            << std::endl;
        return -1;
      }
    }
    return 7;
  }

  void Opset7(OnnxHelper* helper) {
    std::vector<TensorInfo> kernel_info =
        parser_->GetOpInput(block_idx_, op_idx_, "Filter");
    std::vector<TensorInfo> input_info =
        parser_->GetOpInput(block_idx_, op_idx_, "Input");
    std::vector<TensorInfo> output_info =
        parser_->GetOpOutput(block_idx_, op_idx_, "Output");
    auto node =
        helper->MakeNode("Conv", {input_info[0].name, kernel_info[0].name},
                         {output_info[0].name});
    AddAttribute(node, "dilations", dilations_);
    std::vector<int64_t> kernel_shape = {kernel_info[0].shape[2],
                                         kernel_info[0].shape[3]};
    AddAttribute(node, "kernel_shape", kernel_shape);
    AddAttribute(node, "strides", strides_);
    AddAttribute(node, "group", groups_);
    if (padding_algorithm_ == "SAME") {
      AddAttribute(node, "auto_pad", "SAME_UPPER");
    } else if (padding_algorithm_ == "VALID") {
      AddAttribute(node, "auto_pad", "VALID");
    } else {
      AddAttribute(node, "pads", paddings_);
    }
  }

 private:
  std::vector<int64_t> dilations_;
  std::vector<int64_t> strides_;
  std::vector<int64_t> paddings_;
  std::string padding_algorithm_;
  std::string data_format_;
  int64_t groups_;
};

REGISTER_MAPPER(conv2d, Conv2dMapper)
}  // namespace paddle2onnx
