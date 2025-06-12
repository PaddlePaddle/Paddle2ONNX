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

#include "paddle2onnx/mapper/tensor/repeat_interleave.h"
namespace paddle2onnx {
REGISTER_MAPPER(repeat_interleave, RepeatInterleaveMapper)
REGISTER_PIR_MAPPER(repeat_interleave, RepeatInterleaveMapper)

int32_t RepeatInterleaveMapper::GetMinOpsetVersion(bool verbose) {
  constexpr int op_version = 9;
  Logger(verbose, op_version) << RequireOpset(op_version) << std::endl;
  return op_version;
}

void RepeatInterleaveMapper::DynamicRepeatInterleave(
    const std::vector<TensorInfo>& x_info,
    const std::vector<TensorInfo>& out_info) {
  std::string dim_size_name = "";

  auto shape_node = helper_->MakeNode("Shape", {x_info[0].name}, 1);
  auto dim_node =
      helper_->MakeNode("Gather",
                        {shape_node->output(0),
                         helper_->Constant(ONNX_NAMESPACE::TensorProto::INT64,
                                           std::vector<int64_t>{dim_})},
                        1);
  dim_size_name = dim_node->output(0);

  std::string repeat_info_name = "";
  int64_t repeat = 0;
  if (in_pir_mode) {
    if (OpType() == "pd_op.repeat_interleave") {
      GetAttr("repeats", &repeat);
    }
  } else {
    GetAttr("Repeats", &repeat);
  }

  if (HasInput("RepeatTensor")) {
    auto tmp_info = GetInput("RepeatTensor");
    repeat_info_name = helper_->AutoCast(
        tmp_info[0].name, tmp_info[0].dtype, P2ODataType::INT64);
  } else if (repeat != 0) {
    auto repeat_node =
        helper_->MakeNode("Expand",
                          {helper_->Constant(ONNX_NAMESPACE::TensorProto::INT64,
                                             std::vector<int64_t>{repeat}),
                           dim_size_name},
                          1);
    repeat_info_name = repeat_node->output(0);
  }

  std::vector<std::string> split_input_names;

  split_input_names.push_back(x_info[0].name);

  std::vector<std::string> output_names;
  int x_shape_size = x_info[0].shape.size();

  std::string prefix_name = helper_->Constant(
      ONNX_NAMESPACE::TensorProto::INT64, std::vector<int64_t>(dim_, 1));
  std::string suffix_name =
      helper_->Constant(ONNX_NAMESPACE::TensorProto::INT64,
                        std::vector<int64_t>(x_shape_size - dim_ - 1, 1));

  std::string tile_name =
      helper_->Concat({prefix_name, repeat_info_name, suffix_name}, 0);
  auto node = helper_->MakeNode("Tile", {x_info[0].name, tile_name}, 1);
  output_names.push_back(node->output(0));

  helper_->Concat(output_names, out_info[0].name, dim_);
}

void RepeatInterleaveMapper::StaticRepeatInterleave(
    const std::vector<TensorInfo>& x_info,
    const std::vector<TensorInfo>& out_info) {
  int n = x_info[0].shape[dim_];
  int x_shape_size = x_info[0].shape.size();

  std::vector<int64_t> repeats;
  int64_t repeat = 0;
  if (in_pir_mode) {
    if (OpType() == "pd_op.repeat_interleave") {
      GetAttr("repeats", &repeat);
    }
  } else {
    GetAttr("Repeats", &repeat);
  }

  if (repeat != 0) {
    std::vector<int64_t> rp_tmp(n, repeat);
    repeats.assign(rp_tmp.begin(), rp_tmp.end());
  }

  std::string repeat_info_name = "";
  if (HasInput("RepeatTensor")) {
    auto tmp_info = GetInput("RepeatTensor");
    repeat_info_name = helper_->AutoCast(
        tmp_info[0].name, tmp_info[0].dtype, P2ODataType::INT64);
  } else {
    repeat_info_name =
        helper_->Constant(ONNX_NAMESPACE::TensorProto::INT64, repeats);
  }

  std::vector<int64_t> splits(n, 1);

  std::vector<std::string> split_repeat_info_names =
      helper_->Split(repeat_info_name, splits, 0);
  std::vector<std::string> split_input_names =
      helper_->Split(x_info[0].name, splits, dim_);

  int n_suffix_tile = x_shape_size - dim_ - 1;
  int n_prefix_tile = dim_;
  std::string suffix_name =
      helper_->Constant(ONNX_NAMESPACE::TensorProto::INT64,
                        std::vector<int64_t>(n_suffix_tile, 1));
  std::string prefix_name =
      helper_->Constant(ONNX_NAMESPACE::TensorProto::INT64,
                        std::vector<int64_t>(n_prefix_tile, 1));

  std::vector<std::string> output_names;
  for (int i = 0; i < n; i++) {
    std::string tile_name = helper_->Concat(
        {prefix_name, split_repeat_info_names[i], suffix_name}, 0);
    auto node = helper_->MakeNode("Tile", {split_input_names[i], tile_name}, 1);
    output_names.emplace_back(node->output(0));
  }
  helper_->Concat(output_names, out_info[0].name, dim_);
}

void RepeatInterleaveMapper::Opset9() {
  auto x_info = GetInput("X");
  auto out_info = GetOutput("Out");

  bool is_dynamic_shape = false;
  for (auto dim : x_info[0].shape) {
    if (dim == -1) {
      is_dynamic_shape = true;
      break;
    }
  }

  if (is_dynamic_shape) {
    DynamicRepeatInterleave(x_info, out_info);
  } else {
    StaticRepeatInterleave(x_info, out_info);
  }
}
}  // namespace paddle2onnx
