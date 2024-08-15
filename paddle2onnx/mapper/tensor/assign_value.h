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
#include <string>
#include <vector>
#include "paddle2onnx/mapper/mapper.h"
#include <unordered_map>
#include <functional>

namespace paddle2onnx {

class AssignValueMapper : public Mapper {
 public:
  AssignValueMapper(const PaddleParser& p, OnnxHelper* helper, int64_t block_id,
                    int64_t op_id)
      : Mapper(p, helper, block_id, op_id) {
    GetAttr("dtype", &dtype_);
    GetAttr("shape", &shape_);
    GetAttrValues();
  }
  int32_t GetMinOpsetVersion(bool verbose) override;
  void Opset7() override;

 private:
  void GetAttrValues(){
    int32_t dtype = static_cast<int32_t>(dtype_);
    const std::string attr_name = HasAttr("values") ? "values" : GetAttrNameByDtype(dtype);
    std::unordered_map<int32_t, std::function<void()>> type_handlers = {
        {P2ODataType::INT32, [&](){
            if (attr_name == "values") GetScalars(attr_name, &int64_values_);
            else if (attr_name == "int32_values") GetAttr(attr_name, &int64_values_);
        }},
        {P2ODataType::INT64, [&](){
            if (attr_name == "values") GetScalars(attr_name, &int64_values_);
            else if (attr_name == "int64_values") GetAttr(attr_name, &int64_values_);
        }},
        {P2ODataType::FP32, [&](){
            if (attr_name == "values") GetScalars(attr_name, &fp32_values_);
            else if (attr_name == "fp32_values") GetAttr(attr_name, &fp32_values_);
        }},
        {P2ODataType::FP64, [&](){
            if (attr_name == "values") GetScalars(attr_name, &double_values_);
            else if (attr_name == "fp32_values") GetAttr(attr_name, &double_values_);
        }},
        {P2ODataType::BOOL, [&](){
            if (attr_name == "values") GetScalars(attr_name, &bool_values_);
            else if (attr_name == "bool_values") GetAttr(attr_name, &bool_values_);
        }},
    };

    auto handler = type_handlers.find(dtype);
    if (handler != type_handlers.end()) {
        handler->second();
    } else {
        Error() << "Unsupported dtype value" << std::endl;
    }
  }

  std::string GetAttrNameByDtype(int32_t dtype) {
    if (dtype == P2ODataType::INT32) {
      return "int32_values";
    } else if (dtype == P2ODataType::INT64) {
      return "int64_values";
    }else if (dtype == P2ODataType::FP32) {
      return "fp32_values";
    } else if (dtype == P2ODataType::FP64) {
      return "double_values";
    } else if (dtype == P2ODataType::BOOL) {
      return "bool_values";
    }
    Error() << "Unsupported dtype value" << std::endl;

  }

  std::vector<float> fp32_values_;
  std::vector<int64_t> int64_values_;
  std::vector<bool> bool_values_;
  std::vector<double> double_values_;
  std::vector<int32_t> int32_values_;
  std::vector<int64_t> shape_;
  int64_t dtype_;
};

}  // namespace paddle2onnx
