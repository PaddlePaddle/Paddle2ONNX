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

void ReduceLogSumExpMapper::Opset18() {
  GetAttr("keepdim", &keep_dim_);
  GetAttr("reduce_all", &reduce_all_);
  GetAttr("axis", &dim_);

  auto x_info = GetInput("X");
  std::string dims;
  if (!reduce_all_) {
    dims = helper_->Constant(ONNX_NAMESPACE::TensorProto::INT64, dim_);
  } else {
    dims = helper_->Constant(ONNX_NAMESPACE::TensorProto::INT64, Arange(0, x_info[0].Rank()));
  }

  std::string input_name = x_info[0].name;
  auto input_tpye = x_info[0].dtype;
  if (x_info[0].dtype == P2ODataType::BOOL) {
    input_name = helper_->AutoCast(input_name, input_tpye, P2ODataType::INT32);
    input_tpye = P2ODataType::INT32;
  }
  auto reduce_node = helper_->MakeNode("ReduceLogSumExp", {input_name, dims});

  // Add attribute
  AddAttribute(reduce_node, "keepdims", static_cast<int64_t>(keep_dim_));
  auto out_node_name = reduce_node->output(0);

  bool reduce_all_axes = dim_.size() == x_info[0].Rank();
  if (reduce_all_) {
    reduce_all_axes = true;
  }
  if (!keep_dim_ && reduce_all_axes) {
    out_node_name = helper_->Reshape(out_node_name, {-1});
  }
  auto out_info = GetOutput("Out");
  helper_->AutoCast(out_node_name, out_info[0].name, input_tpye, out_info[0].dtype);
}

void ReduceLogSumExpMapper::Opset11() {
  GetAttr("keepdim", &keep_dim_);
  GetAttr("reduce_all", &reduce_all_);
  GetAttr("axis", &dim_);

  auto x_info = GetInput("X");
  auto out_info = GetOutput("Out");
  std::string input_name = x_info[0].name;
  auto input_tpye = x_info[0].dtype;
  if (x_info[0].dtype == P2ODataType::BOOL) {
    input_name = helper_->AutoCast(input_name, input_tpye, P2ODataType::INT32);
    input_tpye = P2ODataType::INT32;
  }
  auto reduce_node = helper_->MakeNode("ReduceLogSumExp", {input_name});

  // Add attribute
  if (!reduce_all_) {
    AddAttribute(reduce_node, "axes", dim_);
  } else {
    AddAttribute(reduce_node, "axes", Arange(0, x_info[0].Rank()));
  }
  AddAttribute(reduce_node, "keepdims", static_cast<int64_t>(keep_dim_));

  auto out = reduce_node->output(0);
  bool reduce_all_axes = dim_.size() == x_info[0].Rank();
  if (reduce_all_) {
    reduce_all_axes = true;
  }
  if (!keep_dim_ && reduce_all_axes) {
    out = helper_->Reshape(out, {-1});
  }
  helper_->AutoCast(out, out_info[0].name, input_tpye, out_info[0].dtype);
}

}  // namespace paddle2onnx
