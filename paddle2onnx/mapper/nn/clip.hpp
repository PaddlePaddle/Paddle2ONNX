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
#include <vector>
#include <iostream>
#include <limits>
#include "paddle2onnx/mapper/mapper.hpp"
#include "paddle2onnx/mapper/mapper_helper.hpp"

namespace paddle2onnx {

class ClipMapper : public Mapper {
 public:
  ClipMapper(const PaddleParser& p, int64_t block_id, int64_t op_id)
      : Mapper(p, block_id, op_id) {}

  int32_t GetMinOpset(bool verbose = false) { return 7; }

  void Opset7(OnnxHelper* helper) {

    auto op = parser->GetOpDesc(block_idx, op_idx);
    bool min_is_tensor = parser->OpHasInput(block_idx, op_idx, "Min");
    bool max_is_tensor = parser->OpHasInput(block_idx, op_idx, "Max");
    Assert(!((min_is_tensor || max_is_tensor) && export_opset_version<11), "min or max of Clip is Tensor, please try opset_version >= 11.");
    
    bool has_min_attr = parser->OpHasAttr(op, "min");
    bool has_max_attr = parser->OpHasAttr(op, "max");
    if(!(min_is_tensor || max_is_tensor)){
      float min = 0.0;
      if (has_min_attr){
        parser->GetOpAttr(op, "min", &min);
      }

      float max = 1.0;
      if (has_max_attr){
        parser->GetOpAttr(op, "max", &max);
      }
      mapper_helper.reset(new ClipHelper(helper, parser, block_idx, op_idx, export_opset_version, has_min_attr, min, has_max_attr, max));
      mapper_helper->Run();
      return;
    }
    
    std::vector<TensorInfo> input_info =
        parser->GetOpInput(block_idx, op_idx, "X");
    std::vector<TensorInfo> output_info =
        parser->GetOpOutput(block_idx, op_idx, "Out");

    int32_t dtype = input_info[0].dtype;
    if (input_info[0].dtype == P2ODataType::FP64){
      dtype = P2ODataType::FP32;
    }
    std::string min_name;
    if (min_is_tensor) {
      std::vector<TensorInfo> min_info =
          parser->GetOpInput(block_idx, op_idx, "Min");
      min_name = helper->AutoCast(min_info[0].name, min_info[0].dtype,
                                  dtype);
    } else {
      float min = 0.0;
      parser->GetOpAttr(op, "min", &min);
      min_name = helper
                     ->MakeConstant({1},
                                    GetOnnxDtype(dtype), min)
                     ->output(0);
    }

    std::string max_name;
    if (max_is_tensor) {
      std::vector<TensorInfo> max_info =
          parser->GetOpInput(block_idx, op_idx, "Max");
      max_name = helper->AutoCast(max_info[0].name, max_info[0].dtype,
                                  dtype);
    } else {
      float max = 1.0;
      parser->GetOpAttr(op, "max", &max);
      max_name = helper
                     ->MakeConstant({1},
                                    GetOnnxDtype(dtype), max)
                     ->output(0);
    }
    if (input_info[0].dtype == P2ODataType::FP64){
      std::string cast_name = helper->AutoCast({input_info[0].name}, P2ODataType::FP64,
                                P2ODataType::FP32);
      std::cout<<"check: "<<cast_name<<" "<<" "<<min_name<<" "<<max_name<<std::endl;
      auto node = helper->MakeNode("Clip", {cast_name, min_name, max_name});
      helper->AutoCast(node->output(0), {output_info[0].name}, P2ODataType::FP32,
                                P2ODataType::FP64);
    }else{
      helper->MakeNode("Clip", {input_info[0].name, min_name, max_name},
                      {output_info[0].name});
    }
  }
};

REGISTER_MAPPER(clip, ClipMapper)
}  // namespace paddle2onnx
