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

#include "paddle2onnx/mapper/quantize/dequantize_linear.h"

namespace paddle2onnx {
REGISTER_MAPPER(dequantize_linear, DequantizeLinearMapper)
REGISTER_PIR_MAPPER(dequantize_linear, DequantizeLinearMapper)

template <typename T>
std::string DequantizeLinearMapper::CreateConstantNode(
    const std::vector<T> &values, ONNX_NAMESPACE::TensorProto_DataType type) {
  if (values.size() == 1) {
    return helper_->Constant({}, type, static_cast<T>(values[0]));
  } else {
    return helper_->Constant(type, values);
  }
}
int32_t DequantizeLinearMapper::GetMinOpsetVersion(bool verbose) { return 19; }

void DequantizeLinearMapper::ConvertInt8ToFp32(
    const std::vector<float> &onnx_scales, std::vector<float> *weight) {
  auto x_info = GetInput("X");
  auto x_shape = x_info[0].shape;
  if (x_shape.size() == 2) {
    for (auto j = 0; j < x_shape[1]; ++j) {
      float scale_value = 0;
      if (onnx_scales.size() == 1) {
        scale_value = onnx_scales[0];
      } else {
        scale_value = onnx_scales[j];
      }
      for (auto i = 0; i < x_shape[0]; ++i) {
        auto offset = i * x_shape[1] + j;
        (*weight)[offset] *= scale_value;
      }
    }
  } else if (x_shape.size() == 4) {
    if (quant_axis_ == 0) {
      auto inner_offset = 1;
      for (auto i : x_shape) {
        inner_offset *= i;
      }
      inner_offset /= x_shape[0];
      for (int i = 0; i < x_shape[0]; ++i) {
        float scale_value = 0;
        if (onnx_scales.size() == 1) {
          scale_value = onnx_scales[0];
        } else {
          scale_value = onnx_scales[i];
        }
        for (auto j = 0; j < inner_offset; ++j) {
          auto offset = i * inner_offset + j;
          (*weight)[offset] *= scale_value;
        }
      }
    } else {
      auto inner_offset = x_shape[2] * x_shape[3];
      auto outter_offset = x_shape[1] * inner_offset;
      for (auto i = 0; i < x_shape[0]; ++i) {
        for (auto j = 0; j < x_shape[1]; ++j) {
          float scale_value = 0;
          if (onnx_scales.size() == 1) {
            scale_value = onnx_scales[0];
          } else {
            scale_value = onnx_scales[j];
          }
          for (auto k = 0; k < inner_offset; k++) {
            auto offset = i * outter_offset + j * inner_offset + k;
            (*weight)[offset] *= scale_value;
          }
        }
      }
    }
  }
}

void DequantizeLinearMapper::Opset19() {
  auto x_info = GetInput("x");
  auto y_info = GetOutput("y");
  auto scale_info = GetInput("scale");
  auto zero_point_info = GetInput("zero_point");

  Assert(qmax_ == 448 || qmax_ == 57344,
         "Paddle2ONNX: Only support e4m3 or e5m2 now.");

  auto input_paddle_dtype = P2ODataType::FLOAT8E4M3FN;
  if (qmax_ == 57344) {
    input_paddle_dtype = P2ODataType::FLOAT8E5M2;
  }
  std::vector<float> denominator_value = {static_cast<float>(qmax_ / 4)};
  std::string denominator_node =
      CreateConstantNode(denominator_value, ONNX_NAMESPACE::TensorProto::FLOAT);

  std::string scale_div_node =
      helper_->MakeNode("Div", {scale_info[0].name, denominator_node})
          ->output(0);

  auto zero_point_node = helper_->AutoCast(
      zero_point_info[0].name, zero_point_info[0].dtype, input_paddle_dtype);

  auto cast_input =
      helper_->AutoCast(x_info[0].name, P2ODataType::FP32, input_paddle_dtype);

  auto DequantizeLinear_node = helper_->MakeNode(
      "DequantizeLinear", {cast_input, scale_div_node, zero_point_node});
  if (helper_->GetOpsetVersion() >= 13) {
    AddAttribute(DequantizeLinear_node, "axis", quant_axis_);
  }

  std::vector<float> qmin_value = {static_cast<float>(qmin_)};
  std::vector<float> qmax_value = {static_cast<float>(qmax_)};

  std::string qmin_node =
      CreateConstantNode(qmin_value, ONNX_NAMESPACE::TensorProto::FLOAT);
  std::string qmax_node =
      CreateConstantNode(qmax_value, ONNX_NAMESPACE::TensorProto::FLOAT);

  std::string min_node =
      helper_->MakeNode("Max", {DequantizeLinear_node->output(0), qmin_node})
          ->output(0);

  std::string final_output =
      helper_->MakeNode("Min", {min_node, qmax_node})->output(0);
  helper_->MakeNode("Identity", {final_output}, {y_info[0].name});
}
}  // namespace paddle2onnx
