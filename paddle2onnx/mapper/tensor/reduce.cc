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

#include "paddle2onnx/mapper/tensor/reduce.h"

namespace paddle2onnx {
REGISTER_MAPPER(reduce_mean, ReduceMapper)
REGISTER_MAPPER(reduce_sum, ReduceMapper)
REGISTER_MAPPER(reduce_min, ReduceMapper)
REGISTER_MAPPER(reduce_max, ReduceMapper)
REGISTER_MAPPER(reduce_prod, ReduceMapper)
REGISTER_MAPPER(logsumexp, ReduceMapper)

void ReduceMapper::Opset7() {
  auto x_info = GetInput("X");
  auto out_info = GetOutput("Out");
  std::map<std::string, std::string> op_map;
  op_map["reduce_mean"] = "ReduceMean";
  op_map["reduce_sum"] = "ReduceSum";
  op_map["reduce_min"] = "ReduceMin";
  op_map["reduce_max"] = "ReduceMax";
  op_map["reduce_prod"] = "ReduceProd";
  op_map["logsumexp"] = "ReduceLogSumExp";
  std::string out = "";
  bool reduce_all_axes = dim_.size() == x_info[0].Rank();
  if (reduce_all_) {
    reduce_all_axes = true;
  }

  if (helper_->GetOpsetVersion() >= 13 && OpType() == "reduce_sum") {
    std::string dims = "";
    if (!reduce_all_) {
      dims = helper_->Constant(ONNX_NAMESPACE::TensorProto::INT64, dim_);
    } else {
      dims = helper_->Constant(ONNX_NAMESPACE::TensorProto::INT64,
                               Arange(0, x_info[0].Rank()));
    }
    auto reduce_node =
        helper_->MakeNode(op_map[OpType()], {x_info[0].name, dims});
    AddAttribute(reduce_node, "keepdims", static_cast<int64_t>(keep_dim_));
    out = reduce_node->output(0);
  } else {
    auto reduce_node = helper_->MakeNode(op_map[OpType()], {x_info[0].name});
    if (!reduce_all_) {
      AddAttribute(reduce_node, "axes", dim_);
    } else {
      AddAttribute(reduce_node, "axes", Arange(0, x_info[0].Rank()));
    }
    AddAttribute(reduce_node, "keepdims", static_cast<int64_t>(keep_dim_));
    out = reduce_node->output(0);
  }
  if (!keep_dim_ && reduce_all_axes) {
    out = helper_->Reshape(out, {-1});
  }
  helper_->AutoCast(out, out_info[0].name, x_info[0].dtype, out_info[0].dtype);
}

}  // namespace paddle2onnx
