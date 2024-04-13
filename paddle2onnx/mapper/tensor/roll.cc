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

#include <limits>
#include "paddle2onnx/mapper/tensor/roll.h"

namespace paddle2onnx {
REGISTER_MAPPER(roll, RollMapper)

void RollMapper::Opset7() {
  auto input_info = GetInput("X");
  auto output_info = GetOutput("Out");

  std::vector<int64_t> shifts;
  GetAttr("shifts", &shifts);

  std::vector<int64_t> axis;
  GetAttr("axis", &axis);

  std::shared_ptr<ONNX_NAMESPACE::NodeProto> temp_node= nullptr;
  auto result_name = input_info[0].name;
  if (axis.empty())
  {
    int64_t axes = 0;
    result_name = helper_->Flatten(result_name);
    for(int i = 0;i < shifts.size();i++) {
      auto shift = shifts[i];
      auto result_0 = helper_->Slice(result_name, {axes}, {-shift}, {(std::numeric_limits<int64_t>::max)()});
      auto result_1 = helper_->Slice(result_name, {axes}, {0}, {-shift});
      temp_node = helper_->MakeNode("Concat", {result_0, result_1});
      AddAttribute(temp_node, "axis", axes);
      result_name = temp_node->output(0);
    }
    helper_->Reshape(result_name, output_info[0].name, input_info[0].shape);
    // helper_->MakeNode("Reshape", {result_name, input_info[0].shape}, {output_info[0].name});
  } else {
    for(int i = 0;i < shifts.size();i++) {
      auto shift = shifts[i];
      int64_t axes = axis[i];
      auto result_0 = helper_->Slice(result_name, {axes}, {-shift}, {(std::numeric_limits<int64_t>::max)()});
      auto result_1 = helper_->Slice(result_name, {axes}, {0}, {-shift});
      if(i+1 == shifts.size()) {
        temp_node = helper_->MakeNode("Concat", {result_0, result_1}, {output_info[0].name});
      } else {
        temp_node = helper_->MakeNode("Concat", {result_0, result_1});
      }
      AddAttribute(temp_node, "axis", axes);
      result_name = temp_node->output(0);
    }
  }
}
}  // namespace paddle2onnx
