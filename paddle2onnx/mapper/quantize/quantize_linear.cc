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

#include "paddle2onnx/mapper/quantize/quantize_linear.h"

namespace paddle2onnx {
REGISTER_MAPPER(quantize_linear, QuantizeLinearMapper)
REGISTER_PIR_MAPPER(quantize_linear, QuantizeLinearMapper)

int32_t QuantizeLinearMapper::GetMinOpsetVersion(bool verbose) { return 19; }

template <typename T>
std::string QuantizeLinearMapper::CreateConstantNode(
    const std::vector<T>& values, ONNX_NAMESPACE::TensorProto_DataType type) {
  if (values.size() == 1) {
    return helper_->Constant({}, type, static_cast<T>(values[0]));
  } else {
    return helper_->Constant(type, values);
  }
}

void QuantizeLinearMapper::Opset19() {
  auto x_info = GetInput("x");
  auto y_info = GetOutput("y");
  auto scale_info = GetInput("scale");
  auto zero_point_info = GetInput("zero_point");

  Assert(qmax_ == 448 || qmax_ == 57344,
         "Paddle2ONNX: Only support e4m3 or e5m2 now.");

  auto output_paddle_dtype = P2ODataType::FLOAT8E4M3FN;
  if (qmax_ == 57344) {
    output_paddle_dtype = P2ODataType::FLOAT8E5M2;
  }

  std::vector<float> denominator_value = {static_cast<float>(qmax_)};
  std::string denominator_node =
      CreateConstantNode(denominator_value, ONNX_NAMESPACE::TensorProto::FLOAT);

  std::string scale_div_node =
      helper_->MakeNode("Div", {scale_info[0].name, denominator_node})
          ->output(0);

  auto zero_point_node = helper_->AutoCast(
      zero_point_info[0].name, zero_point_info[0].dtype, output_paddle_dtype);

  auto QuantizeLinear_node = helper_->MakeNode(
      "QuantizeLinear", {x_info[0].name, scale_div_node, zero_point_node});
  if (helper_->GetOpsetVersion() >= 13) {
    AddAttribute(QuantizeLinear_node, "axis", quant_axis_);
  }
  helper_->AutoCast(QuantizeLinear_node->output(0),
                    y_info[0].name,
                    output_paddle_dtype,
                    P2ODataType::FP32);

  //   QuantizeInfo quantize_info(
  //       onnx_scales, onnx_zeros, onnx_scales, zero_node, quant_axis_);
  //   helper_->quantize_info[x_info[0].name] = quantize_info;
  // }
}
}  // namespace paddle2onnx
