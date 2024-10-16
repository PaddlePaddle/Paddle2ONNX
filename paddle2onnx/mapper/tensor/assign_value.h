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

namespace paddle2onnx {

class AssignValueMapper : public Mapper {
 public:
  AssignValueMapper(const PaddleParser& p, OnnxHelper* helper, int64_t block_id,
                    int64_t op_id)
      : Mapper(p, helper, block_id, op_id) {
    GetAttr("dtype", &dtype_);
    GetAttr("shape", &shape_);
    int32_t dtype = static_cast<int32_t>(dtype_);
    if (dtype == P2ODataType::INT32) {
      GetAttr("int32_values", &int64_values_);
    } else if (dtype == P2ODataType::FP32) {
      GetAttr("fp32_values", &fp32_values_);
    } else if (dtype == P2ODataType::INT64) {
      GetAttr("int64_values", &int64_values_);
    }
  }

  AssignValueMapper(const PaddlePirParser& p, OnnxHelper* helper, int64_t i,
                    bool c)
      : Mapper(p, helper, i, c) {
    in_pir_mode = true;
    // GetAttr("dtype", &dtype_);
    dtype_ = GetOutput("Out")[0].dtype;
    GetAttr("shape", &shape_);
    int32_t dtype = static_cast<int32_t>(dtype_);
    auto array_list = p.global_blocks_ops[i]->attribute("values").dyn_cast<::pir::ArrayAttribute>().AsVector();;
    if (array_list.size() > 0) {
      if(array_list[0].isa<::pir::FloatAttribute>()) {
        auto res = &fp32_values_;
        for (size_t i = 0; i < array_list.size(); ++i) {
          res->push_back(
            array_list[i].dyn_cast<::pir::FloatAttribute>().data());
        }
      } else if (array_list[0].isa<::pir::DoubleAttribute>()) {
        auto res = &fp64_values_;
        for (size_t i = 0; i < array_list.size(); ++i) {
          res->push_back(array_list[i].dyn_cast<::pir::DoubleAttribute>().data());
        }
      } else if (array_list[0].isa<::pir::Int32Attribute>()){
        auto res = &int64_values_;
        for (size_t i = 0; i < array_list.size(); ++i) {
          res->push_back(array_list[i].dyn_cast<::pir::Int32Attribute>().data());
        }
      } else if (array_list[0].isa<::pir::Int64Attribute>()) {
        auto res = &int64_values_;
        for (size_t i = 0; i < array_list.size(); ++i) {
          res->push_back(array_list[i].dyn_cast<::pir::Int64Attribute>().data());
        }
      }
    }
  }
  int32_t GetMinOpsetVersion(bool verbose) override;
  void Opset7() override;

 private:
  std::vector<double> fp64_values_;
  std::vector<float> fp32_values_;
  std::vector<int64_t> int64_values_;
  std::vector<int64_t> shape_;
  int64_t dtype_;
};

}  // namespace paddle2onnx
