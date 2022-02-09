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

class ClipMapper : public Mapper {
 public:
  ClipMapper(const PaddleParser& p, int64_t block_id, int64_t op_id)
      : Mapper(p, block_id, op_id) {}

  int32_t GetMinOpset(bool verbose = false) { return 7; }

  void Opset7(OnnxHelper* helper) {
    std::vector<TensorInfo> input_info =
        parser->GetOpInput(block_idx, op_idx, "X");
    std::vector<TensorInfo> output_info =
        parser->GetOpOutput(block_idx, op_idx, "Out");

    auto op = parser->GetOpDesc(block_idx, op_idx);
    bool min_is_tensor = parser->OpHasInput(block_idx, op_idx, "Min");
    bool max_is_tensor = parser->OpHasInput(block_idx, op_idx, "Max");

    if (!min_is_tensor && !max_is_tensor) {
      float min = 0.0;
      float max = 1.0;
      parser->GetOpAttr(op, "min", &min);
      parser->GetOpAttr(op, "max", &max);
      auto node =
          helper->MakeNode("Clip", {input_info[0].name}, {output_info[0].name});
      AddAttribute(node, "min", min);
      AddAttribute(node, "max", max);
      return;
    }

    std::string min_name;
    if (min_is_tensor) {
      std::vector<TensorInfo> min_info =
          parser->GetOpInput(block_idx, op_idx, "Min");
      min_name = helper->AutoCast(min_info[0].name, min_info[0].dtype,
                                  input_info[0].dtype);
    } else {
      float min = 0.0;
      parser->GetOpAttr(op, "min", &min);
      min_name = helper
                     ->MakeConstant(min_name, {1},
                                    GetOnnxDtype(input_info[0].dtype), min)
                     ->output(0);
    }

    std::string max_name;
    if (max_is_tensor) {
      std::vector<TensorInfo> max_info =
          parser->GetOpInput(block_idx, op_idx, "Max");
      max_name = helper->AutoCast(max_info[0].name, max_info[0].dtype,
                                  input_info[0].dtype);
    } else {
      float max = 1.0;
      parser->GetOpAttr(op, "max", &max);
      max_name = helper
                     ->MakeConstant(max_name, {1},
                                    GetOnnxDtype(input_info[0].dtype), max)
                     ->output(0);
    }

    auto min_node = helper->MakeNode("Min", {input_info[0].name, max_name});
    helper->MakeNode("Max", {min_node->output(0), min_name},
                     {output_info[0].name});
  }

  void Opset11(OnnxHelper* helper) {
    std::vector<TensorInfo> input_info =
        parser->GetOpInput(block_idx, op_idx, "X");
    std::vector<TensorInfo> output_info =
        parser->GetOpOutput(block_idx, op_idx, "Out");

    auto op = parser->GetOpDesc(block_idx, op_idx);
    bool min_is_tensor = parser->OpHasInput(block_idx, op_idx, "Min");
    bool max_is_tensor = parser->OpHasInput(block_idx, op_idx, "Max");

    std::string min_name;
    if (min_is_tensor) {
      std::vector<TensorInfo> min_info =
          parser->GetOpInput(block_idx, op_idx, "Min");
      min_name = helper->AutoCast(min_info[0].name, min_info[0].dtype,
                                  input_info[0].dtype);
    } else {
      float min = 0.0;
      parser->GetOpAttr(op, "min", &min);
      min_name = helper
                     ->MakeConstant(min_name, {1},
                                    GetOnnxDtype(input_info[0].dtype), min)
                     ->output(0);
    }

    std::string max_name;
    if (max_is_tensor) {
      std::vector<TensorInfo> max_info =
          parser->GetOpInput(block_idx, op_idx, "Max");
      min_name = helper->AutoCast(max_info[0].name, max_info[0].dtype,
                                  input_info[0].dtype);
    } else {
      float max = 1.0;
      parser->GetOpAttr(op, "max", &max);
      max_name = helper
                     ->MakeConstant(max_name, {1},
                                    GetOnnxDtype(input_info[0].dtype), max)
                     ->output(0);
    }

    helper->MakeNode("Clip", {input_info[0].name, min_name, max_name},
                     {output_info[0].name});
  }
};

REGISTER_MAPPER(clip, ClipMapper)
}  // namespace paddle2onnx
