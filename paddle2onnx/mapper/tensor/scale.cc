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
#include "paddle2onnx/mapper/tensor/scale.h"

#include <vector>

namespace paddle2onnx {

REGISTER_MAPPER(scale, ScaleMapper)

void ScaleMapper::Opset7() {
  auto input_info = GetInput("X");
  auto output_info = GetOutput("Out");
  bool has_scale_tensor = HasInput("ScaleTensor");
  // TODO(yeliang2258): just temporary use Identity
  bool is_scale_1 = ((scale_ - 1.0) < 1e-06 && (scale_ - 1.0) > -1e-06);
  bool is_bias_0 = (bias_ < 1e-06 && bias_ > -1e-06);

  if (!has_scale_tensor && is_scale_1 && is_bias_0) {
    // TODO(yeliang2258): we could add a pass to eleminate all the identity op
    helper_->MakeNode("Identity", {input_info[0].name}, {output_info[0].name});
  } else {
    // TODO(yeliang2258): we could add a pass to eleminate the scale is 1 or
    // bias is 0
    auto input = helper_->AutoCast(input_info[0].name, input_info[0].dtype,
                                   P2ODataType::FP32);
    std::string out = input;
    if (bias_after_scale_) {
      if (!is_scale_1) {
        auto scale = helper_->Constant(ONNX_NAMESPACE::TensorProto::FLOAT,
                                       std::vector<float>(1, scale_));
        out = helper_->MakeNode("Mul", {out, scale})->output(0);
      }
      if (!is_bias_0) {
        auto bias = helper_->Constant(ONNX_NAMESPACE::TensorProto::FLOAT,
                                      std::vector<float>(1, bias_));
        out = helper_->MakeNode("Add", {out, bias})->output(0);
      }
    } else {
      if (!is_bias_0) {
        auto bias = helper_->Constant(ONNX_NAMESPACE::TensorProto::FLOAT,
                                      std::vector<float>(1, bias_));
        out = helper_->MakeNode("Add", {out, bias})->output(0);
      }
      if (!is_scale_1) {
        auto scale = helper_->Constant(ONNX_NAMESPACE::TensorProto::FLOAT,
                                       std::vector<float>(1, scale_));
        out = helper_->MakeNode("Mul", {out, scale})->output(0);
      }
    }
    helper_->AutoCast(out, output_info[0].name, P2ODataType::FP32,
                      output_info[0].dtype);
  }
}
}  // namespace paddle2onnx
