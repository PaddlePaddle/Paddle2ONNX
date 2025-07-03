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
REGISTER_PIR_MAPPER(reduce_sum, ReduceMapperSum)

int32_t ReduceMapperSum::GetMinOpsetVersion(bool verbose) {
  constexpr int op_version = 13;
  Logger(verbose, op_version) << RequireOpset(op_version) << std::endl;
  return op_version;
}

void ReduceMapperSum::Opset13() {
  auto out_info = GetOutput("Out");
  auto axis_name_ = "dim";
  GetAttr("keep_dim", &keep_dim_);
  if (!in_pir_mode) {
    GetAttr("reduce_all", &reduce_all_);
    GetAttr("in_dtype", &in_dtype_);
    GetAttr("out_dtype", &out_dtype_);
    if (IsAttrVar(axis_name_)) {
      auto info = GetAttrVar(axis_name_);
      TryGetValue(info[0], &dim_);
    } else {
      GetAttr(axis_name_, &dim_);
    }
  } else {
    TryGetInputValue("axis", &dim_);
    // Note: This is a temporary solution. It's needed to be fixed in
    // ProgramTranslator.
    if (dim_.size() == 0 || out_info[0].Rank() == 0 ||
        (out_info[0].Rank() == 1 && out_info[0].shape[0] == 1)) {
      reduce_all_ = true;
    } else {
      reduce_all_ = false;
    }
  }

  auto x_info = GetInput("X");
  auto x_name = x_info[0].name;
  auto x_tpye = x_info[0].dtype;
  if (x_info[0].dtype == P2ODataType::BOOL) {
    x_name = helper_->AutoCast(x_name, x_tpye, P2ODataType::INT32);
    x_tpye = P2ODataType::INT32;
  }

  std::string dims;
  if (IsAttrVar(axis_name_)) {
    auto info = GetAttrVar(axis_name_);
    dims = helper_->AutoCast(info[0].name, info[0].dtype, P2ODataType::INT64);
  } else {
    if (!reduce_all_) {
      dims = helper_->Constant(ONNX_NAMESPACE::TensorProto::INT64, dim_);
    } else {
      dims = helper_->Constant(ONNX_NAMESPACE::TensorProto::INT64,
                               Arange(0, x_info[0].Rank()));
    }
  }

  // Add attribute
  auto reduce_node = helper_->MakeNode("ReduceSum", {x_name, dims});
  AddAttribute(reduce_node, "keepdims", static_cast<int64_t>(keep_dim_));
  auto out_node_name = reduce_node->output(0);

  bool reduce_all_axes = dim_.size() == x_info[0].Rank();
  if (reduce_all_) {
    reduce_all_axes = true;
  }
  if (!keep_dim_ && reduce_all_axes) {
    out_node_name = helper_->Reshape(out_node_name, {-1});
  }
  helper_->AutoCast(out_node_name, out_info[0].name, x_tpye, out_info[0].dtype);
}
}  // namespace paddle2onnx
