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
  
/*
class ClipMapper : public Mapper {
 public:
  ClipMapper(const PaddleParser& p, int64_t block_id, int64_t op_id)
      : Mapper(p, block_id, op_id) {}

  int32_t GetMinOpset(bool verbose = false) { return 7; }

  void Opset7(OnnxHelper* helper) {
    nodes->clear();
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
      auto node = helper->MakeNode("Clip", {input_info[0].name}, {output_info[0].name});
      AddAttribute(node, "min", min);
      AddAttribute(node, "max", max);
      return;
    }

    std::string min_name;
    if (min_is_tensor) {
      std::vector<TensorInfo> min_info =
          parser->GetOpInput(block_idx, op_idx, "Min");
      min_name = AutoCastNode(min_info[0].name, min_info[0].dtype,
                              input_info[0].dtype, nodes);
    } else {
      min_name = MapperHelper::Get()->GenName("clip.min");
      float min = 0.0;
      parser->GetOpAttr(op, "min", &min);
      auto node =
          MakeConstant(min_name, {1}, GetOnnxDtype(input_info[0].dtype), min);
      nodes->push_back(node);
    }

    std::string max_name;
    if (max_is_tensor) {
      std::vector<TensorInfo> max_info =
          parser->GetOpInput(block_idx, op_idx, "Max");
      max_name = AutoCastNode(max_info[0].name, max_info[0].dtype,
                              input_info[0].dtype, nodes);
    } else {
      max_name = MapperHelper::Get()->GenName("clip.max");
      float max = 1.0;
      parser->GetOpAttr(op, "max", &max);
      auto node =
          MakeConstant(max_name, {1}, GetOnnxDtype(input_info[0].dtype), max);
      nodes->push_back(node);
    }

    std::string min_out = MapperHelper::Get()->GenName("cast.minout");
    auto min_node = MakeNode("Min", {input_info[0].name, max_name}, {min_out});
    auto max_node = MakeNode("Max", {min_out, min_name}, {output_info[0].name});
    nodes->push_back(min_node);
    nodes->push_back(max_node);
  }

  void Opset11(std::vector<std::shared_ptr<ONNX_NAMESPACE::NodeProto>>* nodes) {
    nodes->clear();
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
      min_name = AutoCastNode(min_info[0].name, min_info[0].dtype,
                              input_info[0].dtype, nodes);
    } else {
      min_name = MapperHelper::Get()->GenName("clip.min");
      float min = 0.0;
      parser->GetOpAttr(op, "min", &min);
      auto node =
          MakeConstant(min_name, {1}, GetOnnxDtype(input_info[0].dtype), min);
      nodes->push_back(node);
    }

    std::string max_name;
    if (max_is_tensor) {
      std::vector<TensorInfo> max_info =
          parser->GetOpInput(block_idx, op_idx, "Max");
      max_name = AutoCastNode(max_info[0].name, max_info[0].dtype,
                              input_info[0].dtype, nodes);
    } else {
      max_name = MapperHelper::Get()->GenName("clip.max");
      float max = 1.0;
      parser->GetOpAttr(op, "max", &max);
      auto node =
          MakeConstant(max_name, {1}, GetOnnxDtype(input_info[0].dtype), max);
      nodes->push_back(node);
    }

    auto clip_node = MakeNode("Clip", {input_info[0].name, min_name, max_name},
                              {output_info[0].name});
    nodes->push_back(clip_node);
  }
};

REGISTER_MAPPER(clip, ClipMapper)*/
}  // namespace paddle2onnx