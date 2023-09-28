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
  bool is_scale_1 = ((scale_ - 1.0) < 1e-06 && (scale_ - 1.0) > -1e-06);
  bool is_bias_0 = (bias_ < 1e-06 && bias_ > -1e-06);

  if (!has_scale_tensor && is_scale_1 && is_bias_0) {
    helper_->MakeNode("Identity", {input_info[0].name}, {output_info[0].name});
  } else {
    // Scale supports FP32/FP16, so only Y and X aligned data types are required.
    auto p2o_support_type = P2ODataType::FP32;
    auto onnx_support_type = ONNX_NAMESPACE::TensorProto::FLOAT;
    auto input_name = input_info[0].name;
    if(input_info[0].dtype == P2ODataType::FP16) {
      p2o_support_type = P2ODataType::FP16;
      onnx_support_type = ONNX_NAMESPACE::TensorProto::FLOAT16;
    } else {
      input_name = helper_->AutoCast(input_name, input_info[0].dtype, p2o_support_type);
    }
    std::string out = input_name;
    if (bias_after_scale_) {
      if (!is_scale_1 || HasInput("ScaleTensor")) {
        if (HasInput("ScaleTensor")) {
          auto scale_info = GetInput("ScaleTensor");
          auto scale = helper_->AutoCast(scale_info[0].name, scale_info[0].dtype, p2o_support_type);
          out = helper_->MakeNode("Mul", {out, scale})->output(0);
        } else {
          auto scale =
              helper_->Constant({}, onnx_support_type, scale_);
          out = helper_->MakeNode("Mul", {out, scale})->output(0);
        }
      }
      if (!is_bias_0) {
        auto bias =
            helper_->Constant({}, onnx_support_type, bias_);
        out = helper_->MakeNode("Add", {out, bias})->output(0);
      }
    } else {
      if (!is_bias_0) {
        auto bias =
            helper_->Constant({}, onnx_support_type, bias_);
        out = helper_->MakeNode("Add", {out, bias})->output(0);
      }
      if (!is_scale_1 || HasInput("ScaleTensor")) {
        if (HasInput("ScaleTensor")) {
          auto scale_info = GetInput("ScaleTensor");
          auto scale = helper_->AutoCast(
              scale_info[0].name, scale_info[0].dtype, p2o_support_type);
          out = helper_->MakeNode("Mul", {out, scale})->output(0);
        } else {
          auto scale =
              helper_->Constant({}, onnx_support_type, scale_);
          out = helper_->MakeNode("Mul", {out, scale})->output(0);
        }
      }
    }
    helper_->AutoCast(out, output_info[0].name, p2o_support_type, output_info[0].dtype);
  }
}
}  // namespace paddle2onnx
