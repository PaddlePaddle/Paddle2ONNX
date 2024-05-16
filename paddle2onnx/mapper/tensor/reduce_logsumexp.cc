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

#include "paddle2onnx/mapper/tensor/reduce_logsumexp.h"

namespace paddle2onnx {
REGISTER_MAPPER(logsumexp, ReduceLogSumExpMapper)

int32_t ReduceLogSumExpMapper::GetMinOpset(bool verbose) {
  constexpr int op_version = 11;
  Logger(verbose, op_version) << RequireOpset(op_version) << std::endl;
  return op_version;
}

void ReduceLogSumExpMapper::Opset7() {
  auto x_info = GetInput("X");
  auto out_info = GetOutput("Out");
  std::string axis_name = "axis";
  if (IsAttrVar(axis_name)) {
    auto info = GetAttrVar(axis_name);
    TryGetValue(info[0], &dim_);
  } else {
    GetAttr(axis_name, &dim_);
  }

  bool reduce_all_axes = dim_.size() == x_info[0].Rank();
  if (reduce_all_) {
    reduce_all_axes = true;
  }

  std::string input_name = x_info[0].name;
  if (OpType() == "reduce_prod" && x_info[0].dtype == P2ODataType::FP64) {
    input_name = helper_->AutoCast(x_info[0].name, P2ODataType::FP64, P2ODataType::FP32);
  }
  auto reduce_node = helper_->MakeNode("ReduceLogSumExp", {input_name});


  if (!reduce_all_) {
    AddAttribute(reduce_node, "axes", dim_);
  } else {
    AddAttribute(reduce_node, "axes", Arange(0, x_info[0].Rank()));
  }
  AddAttribute(reduce_node, "keepdims", static_cast<int64_t>(keep_dim_));
  auto out = reduce_node->output(0);
  if (OpType() == "reduce_prod" && x_info[0].dtype == P2ODataType::FP64) {
    out = helper_->AutoCast(reduce_node->output(0), P2ODataType::FP32, P2ODataType::FP64);
  }
  if (!keep_dim_ && reduce_all_axes) {
    out = helper_->Reshape(out, {-1});
  }
  helper_->AutoCast(out, out_info[0].name, x_info[0].dtype, out_info[0].dtype);
}

}  // namespace paddle2onnx
