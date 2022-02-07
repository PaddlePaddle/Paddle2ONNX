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
#include "paddle2onnx/mapper/mapper.hpp"

namespace paddle2onnx {

class Conv2dMapper : public Mapper {
 public:
  Conv2dMapper(const PaddleParser& p, int64_t block_id, int64_t op_id)
      : Mapper(p, block_id, op_id) {
    auto op = parser->GetOpDesc(block_idx, op_idx);
    parser->GetOpAttr(op, "groups", &groups);
    parser->GetOpAttr(op, "dilations", &dilations);
    parser->GetOpAttr(op, "strides", &strides);
    parser->GetOpAttr(op, "paddings", &paddings);
    parser->GetOpAttr(op, "padding_algorithm", &padding_algorithm);
    parser->GetOpAttr(op, "data_format", &data_format);
    if (paddings.size() == 2) {
      paddings.push_back(paddings[0]);
      paddings.push_back(paddings[1]);
    } else if (paddings.size() == 4) {
      int32_t tmp = paddings[1];
      paddings[1] = paddings[2];
      paddings[2] = tmp;
    }
  }

  int32_t GetMinOpset(bool verbose = false) {
    if (data_format == "NHWC") {
      if (verbose) {
        std::cerr << "[ERROR] Cannot support NHWC format for operator conv2d."
                  << std::endl;
      }
      return -1;
    }
    return 7;
  }

  void Opset7(std::vector<std::shared_ptr<ONNX_NAMESPACE::NodeProto>>* nodes) {
    nodes->clear();
    std::vector<TensorInfo> kernel_info =
        parser->GetOpInput(block_idx, op_idx, "Filter");
    std::vector<TensorInfo> input_info =
        parser->GetOpInput(block_idx, op_idx, "Input");
    std::vector<TensorInfo> output_info =
        parser->GetOpOutput(block_idx, op_idx, "Output");
    auto node = MakeNode("Conv", {input_info[0].name, kernel_info[0].name},
                         {output_info[0].name});
    AddAttribute(node, "dilations", dilations);
    std::vector<int64_t> kernel_shape = {kernel_info[0].shape[2],
                                         kernel_info[0].shape[3]};
    AddAttribute(node, "kernel_shape", kernel_shape);
    AddAttribute(node, "strides", strides);
    AddAttribute(node, "group", groups);
    if (padding_algorithm == "SAME") {
      AddAttribute(node, "auto_pad", "SAME_UPPER");
    } else if (padding_algorithm == "VALID") {
      AddAttribute(node, "auto_pad", "VALID");
    } else {
      AddAttribute(node, "pads", paddings);
    }
    nodes->push_back(node);
  }

 private:
  std::vector<int64_t> dilations;
  std::vector<int64_t> strides;
  std::vector<int64_t> paddings;
  std::string padding_algorithm;
  std::string data_format;
  int64_t groups;
};

REGISTER_MAPPER(conv2d, Conv2dMapper)
}  // namespace paddle2onnx
