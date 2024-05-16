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

#include "paddle2onnx/mapper/tensor/reduce_min.h"

namespace paddle2onnx {
REGISTER_MAPPER(reduce_min, ReduceMinMapper)
REGISTER_MAPPER(reduce_all, ReduceMinMapper)

int32_t ReduceMinMapper::GetMinOpset(bool verbose) {
  int op_version = 11;

  auto x_info = GetInput("X");
  if (x_info[0].dtype == P2ODataType::FP64) {
    op_version = 12;
  }

  Logger(verbose, op_version) << RequireOpset(op_version) << std::endl;
  return op_version;
}

void ReduceMinMapper::Opset18() {
  GetAttr("keep_dim", &keep_dim_);
  GetAttr("reduce_all", &reduce_all_);
  GetAttr("in_dtype", &in_dtype_);
  GetAttr("out_dtype", &out_dtype_);
  GetAttr("dim", &dim_);

  auto x_info = GetInput("X");
  std::string dims;
  if (!reduce_all_) {
    dims = helper_->Constant(ONNX_NAMESPACE::TensorProto::INT64, dim_);
  } else {
    dims = helper_->Constant(ONNX_NAMESPACE::TensorProto::INT64, Arange(0, x_info[0].Rank()));
  }

  auto input_node_name = x_info[0].name;
  auto input_tpye = x_info[0].dtype;
  if (x_info[0].dtype == P2ODataType::BOOL) {
    input_node_name = helper_->AutoCast(x_info[0].name, x_info[0].dtype, P2ODataType::INT32);
    input_tpye = P2ODataType::INT32;
  }

  // Add attribute
  auto reduce_node = helper_->MakeNode("ReduceMin", {input_node_name, dims});
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

void ReduceMinMapper::Opset12() {
  // The implementation logic of Opset12 is the same as that of Opset11, with the difference being that Opset12 supports input data types as double.
  Opset11();
}

void ReduceMinMapper::Opset11() {
  GetAttr("keep_dim", &keep_dim_);
  GetAttr("reduce_all", &reduce_all_);
  GetAttr("in_dtype", &in_dtype_);
  GetAttr("out_dtype", &out_dtype_);
  GetAttr("dim", &dim_);

  auto x_info = GetInput("X");
  auto input_name = x_info[0].name;
  auto input_tpye = x_info[0].dtype;
  if (x_info[0].dtype == P2ODataType::BOOL) {
    input_name = helper_->AutoCast(input_name, input_tpye, P2ODataType::INT32);
    input_tpye = P2ODataType::INT32;
  }
  auto reduce_node = helper_->MakeNode("ReduceMin", {input_name});

  // Add attribute
  if (!reduce_all_) {
    AddAttribute(reduce_node, "axes", dim_);
  } else {
    AddAttribute(reduce_node, "axes", Arange(0, x_info[0].Rank()));
  }
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
}  // namespace paddle2onnx
