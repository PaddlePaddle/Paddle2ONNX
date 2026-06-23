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

#include "paddle2onnx/mapper/nn/linear_v2.h"

#include <string>
#include <vector>

namespace paddle2onnx {
REGISTER_MAPPER(linear_v2, LinearV2Mapper)
REGISTER_PIR_MAPPER(linear_v2, LinearV2Mapper)

void LinearV2Mapper::Opset7() {
  std::string input_x, input_w, input_b, output_name;
  bool has_bias = false;

  if (in_pir_mode) {
    // PIR: positional inputs/outputs (no named mapping in OpNameNormalizer)
    // Input[0]=X, Input[1]=Weight, Input[2]=Bias, Output[0]=Out
    auto x_info = GetInput(0);
    auto w_info = GetInput(1);
    input_x = x_info[0].name;
    input_w = w_info[0].name;

    // Bias may be a null tensor in PIR - check if the name is valid
    auto bias_info = GetInput(2);
    has_bias = (bias_info.size() > 0 && bias_info[0].name.size() > 0);
    if (has_bias) {
      input_b = bias_info[0].name;
    }

    auto out_info = GetOutput(0);
    output_name = out_info[0].name;
  } else {
    // OLD IR: named inputs/outputs
    auto x_info = GetInput("X");
    auto w_info = GetInput("Weight");
    input_x = x_info[0].name;
    input_w = w_info[0].name;

    if (HasInput("Bias")) {
      auto bias_info = GetInput("Bias");
      has_bias = true;
      input_b = bias_info[0].name;
    }

    auto out_info = GetOutput("Out");
    output_name = out_info[0].name;
  }

  // In PIR format, linear_v2 stores weight in [in_dim, out_dim] layout.
  // When transpose_weight=False (default): y = x @ w (direct matmul)
  // When transpose_weight=True: need to transpose w first
  if (transpose_weight_) {
    // Transpose: [out_dim, in_dim] -> [in_dim, out_dim]
    std::vector<int64_t> perm = {1, 0};
    auto trans_node = helper_->MakeNode("Transpose", {input_w});
    AddAttribute(trans_node, "perm", perm);
    input_w = trans_node->output(0);
  }

  auto matmul = helper_->MakeNode("MatMul", {input_x, input_w});

  if (has_bias) {
    helper_->MakeNode("Add",
                      {matmul->output(0), input_b},
                      {output_name});
  } else {
    helper_->MakeNode("Identity", {matmul->output(0)}, {output_name});
  }
}

}  // namespace paddle2onnx
