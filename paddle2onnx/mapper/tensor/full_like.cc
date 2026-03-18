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
#include "paddle2onnx/mapper/tensor/full_like.h"

namespace paddle2onnx {

REGISTER_PIR_MAPPER(full_like, FullLikeMapper)

void FullLikeMapper::Opset8() {
  auto input_info = GetInput("X");
  auto value_info = GetInput("value");
  auto output_info = GetOutput("Out");

  auto shape_node = helper_->MakeNode("Shape", {input_info[0].name});
  auto expand_node =
      helper_->MakeNode("Expand", {value_info[0].name, shape_node->output(0)});
  helper_->AutoCast(expand_node->output(0),
                    output_info[0].name,
                    value_info[0].dtype,
                    output_info[0].dtype);
}

}  // namespace paddle2onnx
