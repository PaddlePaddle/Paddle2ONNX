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

#include "paddle2onnx/mapper/tensor/reduce_sum.h"

namespace paddle2onnx {
REGISTER_MAPPER(reduce_sum, ReduceMapperSum)

int32_t ReduceMapperSum::GetMinOpset(bool verbose) {
  constexpr int op_version = 13;
  Logger(verbose, op_version) << RequireOpset(op_version) << std::endl;
  return op_version;
}

void ReduceMapperSum::Opset13() {
  auto axis_name_ = "dim";
  GetAttr("keep_dim", &keep_dim_);
  GetAttr("reduce_all", &reduce_all_);
  GetAttr("in_dtype", &in_dtype_);
  GetAttr("out_dtype", &out_dtype_);
  if (IsAttrVar(axis_name_)) {
    auto info = GetAttrVar(axis_name_);
    TryGetValue(info[0], &dim_);
  } else {
    GetAttr(axis_name_, &dim_);
  }

  auto x_info = GetInput("X");
  std::string dims;
  if (IsAttrVar(axis_name_)) {
    auto info = GetAttrVar(axis_name_);
    dims = helper_->AutoCast(info[0].name, info[0].dtype, P2ODataType::INT64);
  } else {
    if (!reduce_all_) {
      dims = helper_->Constant(ONNX_NAMESPACE::TensorProto::INT64, dim_);
    } else {
      dims = helper_->Constant(ONNX_NAMESPACE::TensorProto::INT64, Arange(0, x_info[0].Rank()));
    }
  }

  // Add attribute
  auto reduce_node = helper_->MakeNode("ReduceSum", {x_info[0].name, dims});
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
  helper_->AutoCast(out_node_name, out_info[0].name, x_info[0].dtype, out_info[0].dtype);
}
}  // namespace paddle2onnx
