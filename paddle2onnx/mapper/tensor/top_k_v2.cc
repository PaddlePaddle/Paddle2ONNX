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

#include "paddle2onnx/mapper/tensor/top_k_v2.h"

namespace paddle2onnx {
REGISTER_MAPPER(top_k_v2, TopKMapper)

void TopKMapper::Opset7(OnnxHelper* helper) {
  auto op = parser_->GetOpDesc(block_idx_, op_idx_);
  auto x_info = GetInput("X");
  auto output_info = GetOutput("Out");
  auto indices_info = GetOutput("Indices");

  std::string k = "";
  if (HasInput("K")) {
    auto k_info = GetInput("K");
    k = helper->AutoCast(k_info[0].name, k_info[0].dtype, P2ODataType::INT64);
    if (k_info[0].Rank() == 0) {
      k = helper->Reshape(k, std::vector<int64_t>(1, -1));
    }
  } else {
    int64_t k_value = 0;
    GetAttr("k", &k_value);
    k = helper->Constant({1}, ONNX_NAMESPACE::TensorProto::INT64, k_value);
  }
  auto out_node = helper->MakeNode("TopK", {x_info[0].name, k}, 2);
  AddAttribute(out_node, "largest", static_cast<int64_t>(largest_));
  AddAttribute(out_node, "sorted", static_cast<int64_t>(sorted_));
  AddAttribute(out_node, "axis", axis_);
  helper->AutoCast(out_node->output(0), output_info[0].name, x_info[0].dtype,
                   output_info[0].dtype);
  helper->AutoCast(out_node->output(1), indices_info[0].name,
                   P2ODataType::INT64, indices_info[0].dtype);
}

}  // namespace paddle2onnx
