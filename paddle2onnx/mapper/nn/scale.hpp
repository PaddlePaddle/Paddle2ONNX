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
#pragma once
#include "paddle2onnx/mapper/mapper.hpp"

namespace paddle2onnx {

class ScaleMapper : public Mapper {
 public:
  ScaleMapper(const PaddleParser& p, int64_t block_id, int64_t op_id)
      : Mapper(p, block_id, op_id) {
    auto op = parser->GetOpDesc(block_idx, op_idx);
    parser->GetOpAttr(op, "scale", &scale);
    parser->GetOpAttr(op, "bias", &bias);
    parser->GetOpAttr(op, "bias_after_scale", &bias_after_scale);
  }

  int32_t GetMinOpset(bool verbose = false) { return 7; }

  void Opset7(OnnxHelper* helper) {
    std::vector<TensorInfo> input_info =
        parser->GetOpInput(block_idx, op_idx, "X");
    std::vector<TensorInfo> output_info =
        parser->GetOpOutput(block_idx, op_idx, "Out");
    // TODO just temporary use Identity
    bool is_scale_1 = ((scale - 1.0) < 1e-06 && (scale - 1.0) > -1e-06);
    bool is_bias_0 = (bias < 1e-06 && bias > -1e-06);
    if (is_scale_1 && is_bias_0) {
      // TODO we could add a pass to eleminate all the identity op
      helper->MakeNode("Identity", {input_info[0].name}, {output_info[0].name});
    } else {
      // TODO we could add a pass to eleminate the scale is 1 or bias is 0
      auto onnx_dtype = GetOnnxDtype(input_info[0].dtype);
      auto bias_node = helper->MakeConstant({1}, onnx_dtype, bias);
      auto scale_node = helper->MakeConstant({1}, onnx_dtype, scale);
      std::string scale_name = scale_node->output(0);
      std::string bias_name = bias_node->output(0);
      if (bias_after_scale) {
        auto mul_node = helper->MakeNode("Mul", {input_info[0].name, scale_name});
        std::string mul_out = mul_node->output(0);
        helper->MakeNode("Mul", {input_info[0].name, scale_name}, {mul_out});
        helper->MakeNode("Add", {mul_out, bias_name}, {output_info[0].name});
      } else {
        auto add_node = helper->MakeNode("Add", {input_info[0].name, bias_name});
        std::string add_out = add_node->output(0);
        helper->MakeNode("Mul", {add_out, scale_name}, {output_info[0].name});
      }
    }
  }

 private:
  float scale = 1.0;
  float bias = 0.0;
  bool bias_after_scale = true;
};

REGISTER_MAPPER(scale, ScaleMapper)
}  // namespace paddle2onnx
