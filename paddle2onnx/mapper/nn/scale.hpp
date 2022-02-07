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

  void Opset7(std::vector<std::shared_ptr<ONNX_NAMESPACE::NodeProto>>* nodes) {
    nodes->clear();
    std::vector<TensorInfo> input_info =
        parser->GetOpInput(block_idx, op_idx, "X");
    std::vector<TensorInfo> output_info =
        parser->GetOpOutput(block_idx, op_idx, "Out");
    // TODO just temporary use Identity
    bool is_scale_1 = ((scale - 1.0) < 1e-06 && (scale - 1.0) > -1e-06);
    bool is_bias_0 = (bias < 1e-06 && bias > -1e-06);
    if (is_scale_1 && is_bias_0) {
      auto node =
          MakeNode("Identity", {input_info[0].name}, {output_info[0].name});
      nodes->push_back(node);
    } else {
      // TODO we could add a pass to eleminate the scale is 1 or bias is 0
      auto onnx_dtype = GetOnnxDtype(input_info[0].dtype);
      auto bias_name = MapperHelper::Get()->GenName("scale.bias");
      auto scale_name = MapperHelper::Get()->GenName("scale.scale");
      auto bias_node = MakeConstant(bias_name, {1}, onnx_dtype, bias);
      auto scale_node = MakeConstant(scale_name, {1}, onnx_dtype, scale);
      nodes->push_back(bias_node);
      nodes->push_back(scale_node);
      if (bias_after_scale) {
        auto mul_out = MapperHelper::Get()->GenName("scale.mul");
        auto mul_node =
            MakeNode("Mul", {input_info[0].name, scale_name}, {mul_out});
        auto add_node =
            MakeNode("Add", {mul_out, bias_name}, {output_info[0].name});
        nodes->push_back(mul_node);
        nodes->push_back(add_node);
      } else {
        auto add_out = MapperHelper::Get()->GenName("scale.add");
        auto add_node =
            MakeNode("Add", {input_info[0].name, bias_name}, {add_out});
        auto mul_node =
            MakeNode("Mul", {add_out, scale_name}, {output_info[0].name});
        nodes->push_back(add_node);
        nodes->push_back(mul_node);
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
