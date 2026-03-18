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
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

#include "paddle2onnx/mapper/mapper.h"

namespace paddle2onnx {

class AssignValueMapper : public Mapper {
 public:
  AssignValueMapper(const PaddleParser& p,
                    OnnxHelper* helper,
                    int64_t block_id,
                    int64_t op_id)
      : Mapper(p, helper, block_id, op_id) {
    GetAttr("dtype", &dtype_);
    GetAttr("shape", &shape_);
    GetAttrValues();
  }

  AssignValueMapper(const PaddlePirParser& p,
                    OnnxHelper* helper,
                    int64_t op_id,
                    bool in_cf_block)
      : Mapper(p, helper, op_id, in_cf_block) {
    in_pir_mode = true;
    GetAttr("dtype", &dtype_);
    GetAttr("shape", &shape_);
    int32_t dtype = static_cast<int32_t>(dtype_);
    pir::Operation* op = if_in_cf_block ? p.sub_blocks_ops[pir_op_idx_]
                                        : p.global_blocks_ops[pir_op_idx_];
    auto array_list =
        op->attribute("values").dyn_cast<::pir::ArrayAttribute>().AsVector();
    if (array_list.size() > 0) {
      if (array_list[0].isa<::pir::FloatAttribute>()) {
        auto res = &fp32_values_;
        for (size_t i = 0; i < array_list.size(); ++i) {
          res->push_back(
              array_list[i].dyn_cast<::pir::FloatAttribute>().data());
        }
      } else if (array_list[0].isa<::pir::DoubleAttribute>()) {
        auto res = &fp64_values_;
        for (size_t i = 0; i < array_list.size(); ++i) {
          res->push_back(
              array_list[i].dyn_cast<::pir::DoubleAttribute>().data());
        }
      } else if (array_list[0].isa<::pir::Int32Attribute>()) {
        auto res = &int64_values_;
        for (size_t i = 0; i < array_list.size(); ++i) {
          res->push_back(
              array_list[i].dyn_cast<::pir::Int32Attribute>().data());
        }
      } else if (array_list[0].isa<::pir::Int64Attribute>()) {
        auto res = &int64_values_;
        for (size_t i = 0; i < array_list.size(); ++i) {
          res->push_back(
              array_list[i].dyn_cast<::pir::Int64Attribute>().data());
        }
      }
    }
  }
  int32_t GetMinOpsetVersion(bool verbose) override { return 7; };
  void Opset7() override;

 private:
  void GetAttrValues() {
    int32_t dtype = static_cast<int32_t>(dtype_);
    const std::string attr_name =
        HasAttr("values") ? "values" : GetAttrNameByDtype(dtype);
    std::unordered_map<int32_t, std::function<void()>> type_handlers = {
        {P2ODataType::INT32,
         [&]() {
           if (attr_name == "values")
             GetScalars(attr_name, &int64_values_);
           else if (attr_name == GetAttrNameByDtype(dtype_))
             GetAttr(attr_name, &int64_values_);
         }},
        {P2ODataType::INT64,
         [&]() {
           if (attr_name == "values")
             GetScalars(attr_name, &int64_values_);
           else if (attr_name == GetAttrNameByDtype(dtype_))
             GetAttr(attr_name, &int64_values_);
         }},
        {P2ODataType::FP32,
         [&]() {
           if (attr_name == "values")
             GetScalars(attr_name, &fp32_values_);
           else if (attr_name == GetAttrNameByDtype(dtype_))
             GetAttr(attr_name, &fp32_values_);
         }},
        {P2ODataType::FP64,
         [&]() {
           if (attr_name == "values")
             GetScalars(attr_name, &fp64_values_);
           else if (attr_name == GetAttrNameByDtype(dtype_))
             GetAttr(attr_name, &fp64_values_);
         }},
    };

    auto handler = type_handlers.find(dtype);
    Assert(handler != type_handlers.end(), "Unsupported dtype value");
    handler->second();
  }

  std::string GetAttrNameByDtype(int32_t dtype) {
    if (dtype == P2ODataType::INT32) {
      return "int32_values";
    } else if (dtype == P2ODataType::INT64) {
      return "int64_values";
    } else if (dtype == P2ODataType::FP32) {
      return "fp32_values";
    } else if (dtype == P2ODataType::FP64) {
      return "double_values";
    }
    Assert(false, "Only supports int32/int64/fp32/fp64.");
    return "";
  }

  std::vector<int64_t> int64_values_;
  std::vector<float> fp32_values_;
  std::vector<double> fp64_values_;
  std::vector<int64_t> shape_;
  int64_t dtype_;
};

}  // namespace paddle2onnx
