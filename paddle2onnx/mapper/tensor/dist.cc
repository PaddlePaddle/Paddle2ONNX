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

#include "paddle2onnx/mapper/tensor/dist.h"

#include <vector>
#include <cmath>

namespace paddle2onnx {

REGISTER_MAPPER(dist, DistMapper)

int __NaN=0xFFC00000,__Infinity=0x7F800000,__Neg_Infinity=0xFF800000;
const float NaN=*((float *)&__NaN),Infinity=*((float *)&__Infinity),Neg_Infinity=*((float *)&__Neg_Infinity);

int32_t DistMapper::GetMinOpset(bool verbose) {
  if (fabs(p_) < 1e-6) {
    Logger(true, 9) << "While p is 0.0, "<< RequireOpset(9) << std::endl;
    return 9;
  }
  return 7;
}

void DistMapper::Opset7() {
  auto x_info = GetInput("X");
  auto y_info = GetInput("Y");
  auto output_info = GetOutput("Out");

  auto sub_node = helper_->MakeNode("Sub", {x_info[0].name, y_info[0].name});
  auto abs_node = helper_->MakeNode("Abs", {sub_node->output(0)});

  if (fabs(p_) < 1e-6) {
      auto sign_node = helper_->MakeNode("Sign", {abs_node->output(0)});
      auto sum_node = helper_->MakeNode("ReduceSum", {sign_node->output(0)});
      AddAttribute(sum_node, "keepdims", static_cast<int64_t>(0));
      auto s_sum_node = helper_->Reshape(sum_node->output(0), {-1});
      helper_->AutoCast(s_sum_node, output_info[0].name, x_info[0].dtype, output_info[0].dtype);
  } else if (p_ == Infinity) {
      auto max_node = helper_->MakeNode("ReduceMax", {abs_node->output(0)});
      AddAttribute(max_node, "keepdims", static_cast<int64_t>(0));
      auto s_max_node = helper_->Reshape(max_node->output(0), {-1});
      helper_->AutoCast(s_max_node, output_info[0].name, x_info[0].dtype, output_info[0].dtype);
  } else if (p_ == Neg_Infinity) {
      auto min_node = helper_->MakeNode("ReduceMin", {abs_node->output(0)});
      AddAttribute(min_node, "keepdims", static_cast<int64_t>(0));
      auto s_min_node = helper_->Reshape(min_node->output(0), {-1});
      helper_->AutoCast(s_min_node, output_info[0].name, x_info[0].dtype, output_info[0].dtype);
  } else {
      std::string p = helper_->Constant({1}, GetOnnxDtype(x_info[0].dtype), p_);
      auto pow_node = helper_->MakeNode("Pow", {abs_node->output(0), p});

      auto sum_node = helper_->MakeNode("ReduceSum", {pow_node->output(0)});
      AddAttribute(sum_node, "keepdims", static_cast<int64_t>(0));
      auto s_node = helper_->Reshape(sum_node->output(0), {-1});

      auto p_1 = helper_->MakeNode("Reciprocal", {p});
      helper_->MakeNode("Pow", {s_node, p_1->output(0)}, {output_info[0].name});
  }
}

}  // namespace paddle2onnx
