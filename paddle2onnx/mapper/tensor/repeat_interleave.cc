
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

int32_t RepeatInterleaveMapper::GetMinOpsetVersion(bool verbose) {
  constexpr int op_version = 9;
  Logger(verbose, op_version) << RequireOpset(op_version) << std::endl;
  return op_version;
}

void RepeatInterleaveMapper::Opset9() {
  auto x_info = GetInput("X");  // shape = [1, 2, 3]
  auto out_info = GetOutput("Out");
  int n = x_info[0].shape[dim_];
  int x_shape_size = x_info[0].shape.size();

  std::vector<int64_t> repeats;
  int64_t repeat;
  GetAttr("Repeats", &repeat);
  if (repeat != 0) {
    std::vector<int64_t> rp_tmp(n, repeat);
    repeats.assign(rp_tmp.begin(), rp_tmp.end());
  }

  std::string repeat_info_name = "";
  if (HasInput("RepeatsTensor")) {
    auto tmp_info = GetInput("RepeatsTensor");
    repeat_info_name = helper_->AutoCast(tmp_info[0].name, tmp_info[0].dtype,
                                         P2ODataType::INT64);
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
}  // namespace paddle2onnx