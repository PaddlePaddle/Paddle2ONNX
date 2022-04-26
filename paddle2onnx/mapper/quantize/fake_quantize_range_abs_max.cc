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

#include "paddle2onnx/mapper/quantize/fake_quantize_range_abs_max.h"

namespace paddle2onnx {
REGISTER_MAPPER(fake_quantize_range_abs_max, FakeQuantizeRangeAbsMaxMapper)

int32_t FakeQuantizeRangeAbsMaxMapper::GetMinOpset(bool verbose) {
  if (!IsConstantInput("InScale")) {
    Error() << "Input `InScale` requires to be a constant tensor." << std::endl;
    return -1;
  }
  std::vector<float> scales;
  if (!TryGetInputValue("InScale", &scales)) {
    Error() << "Failed to read tensor value of `InScale`." << std::endl;
    return -1;
  }
  if (scales.size() > 1) {
    Error() << "Only support InScale.size = 1." << std::endl;
    return -1;
  }
  if (bit_length_ != 8) {
    Error() << "Only support bit_length = 8." << std::endl;
    return -1;
  }
  Logger(verbose, 10) << RequireOpset(10) << std::endl;
  return 10;
}

void FakeQuantizeRangeAbsMaxMapper::Opset10() {
  auto x_info = GetInput("X");
  std::vector<float> scales;
  Assert(TryGetInputValue("InScale", &scales),
         "Failed to read tensor value of `InScale`.");
  auto cliped_input =
      helper_->Clip(x_info[0].name, -1 * scales[0], scales[0], x_info[0].dtype);
  float y_scale_value = scales[0] / 127;
  auto y_scale =
      helper_->Constant({}, ONNX_NAMESPACE::TensorProto::FLOAT, y_scale_value);
  auto y_zero_point =
      helper_->Constant({}, ONNX_NAMESPACE::TensorProto::INT8, int(0));
  auto quant_out =
      helper_->MakeNode("QuantizeLinear", {cliped_input, y_scale, y_zero_point})
          ->output(0);

  auto x_zero_point =
      helper_->Constant({}, ONNX_NAMESPACE::TensorProto::INT8, int(0));
  auto x_scale =
      helper_->Constant({}, ONNX_NAMESPACE::TensorProto::FLOAT, y_scale_value);
  auto dequant_out =
      helper_->MakeNode("DequantizeLinear", {quant_out, x_scale, x_zero_point},
                        {GetOutput("Out")[0].name});
}
}  // namespace paddle2onnx
