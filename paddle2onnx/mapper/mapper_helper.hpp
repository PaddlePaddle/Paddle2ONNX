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
#include "paddle2onnx/mapper/mapper.hpp"

namespace paddle2onnx {

class ClipHelper : public Base_mapper_helper {
 public:
  ClipHelper(OnnxHelper* helper, const PaddleParser* parser, const int64_t& block_id,
  const int64_t& op_id, const int32_t& export_opset_version, const bool& has_min_attr, const float& min, const bool& has_max_attr, const float& max) {
    helper_ = helper;
    parser_ = parser;
    has_min_attr_ = has_min_attr;
    has_max_attr_ = has_max_attr;
    min_ = min;
    max_ = max;
    block_id_ = block_id;
    op_id_ = op_id;
    export_opset_version_ = export_opset_version;
  }

  virtual void Run(){
      std::vector<TensorInfo> input_info =
        parser_->GetOpInput(block_id_, op_id_, "X");
      std::vector<TensorInfo> output_info =
          parser_->GetOpOutput(block_id_, op_id_, "Out");

      std::string input_name;
      if (input_info[0].dtype == P2ODataType::FP64){
        input_name = helper_->AutoCast(input_info[0].name, input_info[0].dtype,
                                  P2ODataType::FP32);
      }else{
        input_name = input_info[0].name;
      }

      if (export_opset_version_ < 11){
        if (input_info[0].dtype == P2ODataType::FP64){
          auto node =
              helper_->MakeNode("Clip", {input_name});
          if (has_max_attr_){
            AddAttribute(node, "max", max_);
          }
          if (has_min_attr_){
            AddAttribute(node, "min", min_);
          }
          helper_->AutoCast(node->output(0), output_info[0].name, P2ODataType::FP32,
                                P2ODataType::FP64);
        }else{
          auto node =
              helper_->MakeNode("Clip", {input_name}, {output_info[0].name});
          if (has_max_attr_){
            AddAttribute(node, "max", max_);
          }
          if (has_min_attr_){
            AddAttribute(node, "min", min_);
          }
        } 
      }else{
        if (input_info[0].dtype == P2ODataType::FP64){
          std::string min_name;
          int32_t dtype = P2ODataType::FP32;
          min_name = helper_->MakeConstant({1},
                                      GetOnnxDtype(dtype), min_)->output(0);
          std::string max_name;
          max_name = helper_
                      ->MakeConstant({1},
                                      GetOnnxDtype(dtype), max_)->output(0);
          auto node = helper_->MakeNode("Clip", {input_name, min_name, max_name});
          helper_->AutoCast(node->output(0), {output_info[0].name}, P2ODataType::FP32,
                                    P2ODataType::FP64);
        }else{
          std::string min_name;
          int32_t dtype = input_info[0].dtype;
          min_name = helper_->MakeConstant({1},
                                      GetOnnxDtype(dtype), min_)->output(0);
          std::string max_name;
          max_name = helper_
                      ->MakeConstant({1},
                                      GetOnnxDtype(dtype), max_)->output(0);
          auto node = helper_->MakeNode("Clip", {input_name, min_name, max_name}, {output_info[0].name});
        }
      } 
  }

 private:
  OnnxHelper* helper_;
  const PaddleParser* parser_;
  int64_t block_id_;
  int64_t op_id_;
  float min_;
  float max_;
  bool has_min_attr_ = false;
  bool has_max_attr_ = false;
  int32_t export_opset_version_;
};

}