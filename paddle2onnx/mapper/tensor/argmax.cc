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

#include "paddle2onnx/mapper/tensor/argmax.h"

namespace paddle2onnx {
REGISTER_MAPPER(arg_max, ArgMaxMapper)

void ArgMaxMapper::Opset7() {
  auto input_info = parser_->GetOpInput(block_idx_, op_idx_, "X");
  auto output_info = parser_->GetOpOutput(block_idx_, op_idx_, "Out");
  auto input = input_info[0].name;
  if (flatten_) {
    input = helper_->Flatten(input_info[0].name);
  }
  auto arg_node = helper_->MakeNode("ArgMax", {input});
  AddAttribute(arg_node, "axis", axis_);
  AddAttribute(arg_node, "keepdims", static_cast<int64_t>(keepdims_));
  helper_->AutoCast(arg_node->output(0), output_info[0].name,
                    P2ODataType::INT64, output_info[0].dtype);
}

}  // namespace paddle2onnx
