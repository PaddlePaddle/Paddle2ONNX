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
#include "paddle2onnx/mapper/nn/scale.h"
#include <vector>

namespace paddle2onnx {

REGISTER_MAPPER(scale, ScaleMapper)

void ScaleMapper::Opset7(OnnxHelper* helper) {
  std::vector<TensorInfo> input_info =
      parser_->GetOpInput(block_idx_, op_idx_, "X");
  std::vector<TensorInfo> output_info =
      parser_->GetOpOutput(block_idx_, op_idx_, "Out");
  // TODO(yeliang2258): just temporary use Identity
  bool is_scale_1 = ((scale_ - 1.0) < 1e-06 && (scale_ - 1.0) > -1e-06);
  bool is_bias_0 = (bias_ < 1e-06 && bias_ > -1e-06);
  if (is_scale_1 && is_bias_0) {
    // TODO(yeliang2258): we could add a pass to eleminate all the identity op
    helper->MakeNode("Identity", {input_info[0].name}, {output_info[0].name});
  } else {
    // TODO(yeliang2258): we could add a pass to eleminate the scale is 1 or
    // bias is 0
    auto onnx_dtype = GetOnnxDtype(input_info[0].dtype);
    auto bias_node = helper->MakeConstant({1}, onnx_dtype, bias_);
    auto scale_node = helper->MakeConstant({1}, onnx_dtype, scale_);
    if (bias_after_scale_) {
      auto mul_node =
          helper->MakeNode("Mul", {input_info[0].name, scale_node->output(0)});
      helper->MakeNode("Add", {mul_node->output(0), bias_node->output(0)},
                       {output_info[0].name});
    } else {
      auto add_node =
          helper->MakeNode("Add", {input_info[0].name, bias_node->output(0)});
      helper->MakeNode("Mul", {add_node->output(0), scale_node->output(0)},
                       {output_info[0].name});
    }
  }
}
}  // namespace paddle2onnx
