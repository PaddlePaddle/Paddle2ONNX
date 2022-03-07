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

#include "paddle2onnx/mapper/tensor/lookup_table.h"
#include <iostream>
#include <string>
#include <vector>

namespace paddle2onnx {
REGISTER_MAPPER(lookup_table, LookupTableMapper)
REGISTER_MAPPER(lookup_table_v2, LookupTableMapper)

int32_t LookupTableMapper::GetMinOpset(bool verbose) {
  std::vector<TensorInfo> input_w_info =
      parser_->GetOpInput(block_idx_, op_idx_, "W");
  bool has_minus = false;
  for (auto i : input_w_info[0].shape) {
    has_minus = (i == -1);
    if (has_minus) {
      break;
    }
  }
  if (padding_idx_ != -1 && has_minus) {
    return 11;
  }
  return 7;
}

void LookupTableMapper::Opset7(OnnxHelper* helper) {
  auto op = parser_->GetOpDesc(block_idx_, op_idx_);
  std::vector<TensorInfo> input_ids_info =
      parser_->GetOpInput(block_idx_, op_idx_, "Ids");
  std::vector<TensorInfo> input_w_info =
      parser_->GetOpInput(block_idx_, op_idx_, "W");
  std::vector<TensorInfo> output_info =
      parser_->GetOpOutput(block_idx_, op_idx_, "Out");

  std::string ids_node = input_ids_info[0].name;
  auto ids_shape = input_ids_info[0].shape;
  if (op.type() == "lookup_table" && ids_shape[ids_shape.size() - 1] == 1) {
    ids_node = helper->Squeeze(input_ids_info[0].name, {-1});
  }

  auto input_shape = input_w_info[0].shape;
  int64_t sum_val = 1;
  for (auto i : input_shape) {
    sum_val *= i;
  }
  int interval = sum_val / input_shape[0];

  if (padding_idx_ != -1) {
    std::vector<int64_t> data(sum_val, 1);
    for (auto i = 0; i < interval; i++) {
      data[padding_idx_ * interval + i] = 0;
    }
    std::string constant = helper->Constant(
        input_shape, GetOnnxDtype(input_w_info[0].dtype), data);
    auto weight_node =
        helper->MakeNode("Mul", {input_w_info[0].name, constant});
    helper->MakeNode("Gather", {weight_node->output(0), ids_node},
                     {output_info[0].name});
  } else {
    helper->MakeNode("Gather", {input_w_info[0].name, ids_node},
                     {output_info[0].name});
  }
}

void LookupTableMapper::Opset11(OnnxHelper* helper) {
  auto op = parser_->GetOpDesc(block_idx_, op_idx_);
  std::vector<TensorInfo> input_ids_info =
      parser_->GetOpInput(block_idx_, op_idx_, "Ids");
  std::vector<TensorInfo> input_w_info =
      parser_->GetOpInput(block_idx_, op_idx_, "W");
  std::vector<TensorInfo> output_info =
      parser_->GetOpOutput(block_idx_, op_idx_, "Out");

  std::string ids_node = input_ids_info[0].name;
  auto ids_shape = input_ids_info[0].shape;
  if (op.type() == "lookup_table" && ids_shape[ids_shape.size() - 1] == 1) {
    ids_node = helper->Squeeze(input_ids_info[0].name, {-1});
  }

  auto input_shape = input_w_info[0].shape;
  int64_t sum_val = 1;
  for (auto i : input_shape) {
    sum_val *= i;
  }
  int interval = sum_val / input_shape[0];

  if (padding_idx_ != -1) {
    bool has_minus = false;
    for (auto i : input_w_info[0].shape) {
      has_minus = (i == -1);
      if (has_minus) {
        break;
      }
    }
    if (has_minus) {
      std::vector<int64_t> shape = {interval};
      std::string replace_data =
          helper->Constant(shape, GetOnnxDtype(input_w_info[0].dtype), 0.0);
      std::string index = helper->Constant(
          {1}, ONNX_NAMESPACE::TensorProto::INT64, padding_idx_);
      auto scatter_node = helper->MakeNode(
          "ScatterND", {input_w_info[0].name, index, replace_data});
      helper->MakeNode("Gather", {scatter_node->output(0), ids_node},
                       {output_info[0].name});
    } else {
      std::vector<int64_t> data(sum_val, 1);
      for (auto i = 0; i < interval; i++) {
        data[padding_idx_ * interval + i] = 0;
      }
      std::string constant = helper->Constant(
          input_shape, GetOnnxDtype(input_w_info[0].dtype), data);
      auto weight_node =
          helper->MakeNode("Mul", {input_w_info[0].name, constant});
      helper->MakeNode("Gather", {weight_node->output(0), ids_node},
                       {output_info[0].name});
    }
  } else {
    helper->MakeNode("Gather", {input_w_info[0].name, ids_node},
                     {output_info[0].name});
  }
}

}  // namespace paddle2onnx
