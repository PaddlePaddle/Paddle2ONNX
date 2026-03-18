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

class SetValueMapper : public Mapper {
 public:
  SetValueMapper(const PaddleParser& p,
                 OnnxHelper* helper,
                 int64_t block_id,
                 int64_t op_id)
      : Mapper(p, helper, block_id, op_id) {
    MarkAsExperimentalOp();
    GetAttr("axes", &axes_);
    GetAttr("starts", &starts_);
    GetAttr("ends", &ends_);
    GetAttr("steps", &steps_);
    GetAttr("shape", &shape_);
    GetAttr("decrease_axes", &decrease_axes_);
    GetAttr("none_axes", &none_axes_);
    if (!HasInput("ValueTensor")) {
      auto dtype = GetInput("Input")[0].dtype;
      if (dtype == P2ODataType::INT32) {
        GetAttr("int32_values", &int_values_);
      } else if (dtype == P2ODataType::INT64) {
        GetAttr("int64_values", &int_values_);
      } else if (dtype == P2ODataType::FP32) {
        GetAttr("fp32_values", &fp32_values_);
      } else if (dtype == P2ODataType::FP64) {
        GetAttr("fp64_values", &fp64_values_);
      }
    }
  }
  SetValueMapper(const PaddlePirParser& p,
                 OnnxHelper* helper,
                 int64_t op_id,
                 bool if_in_cf_block)
      : Mapper(p, helper, op_id, if_in_cf_block) {
    MarkAsExperimentalOp();
    GetAttr("axes", &axes_);
    GetAttr("decrease_axes", &decrease_axes_);
    GetAttr("none_axes", &none_axes_);
    if (HasAttr("shape")) GetAttr("shape", &shape_);
    if (HasAttr("values")) {
      pir::Operation* op = if_in_cf_block ? p.sub_blocks_ops[pir_op_idx_]
                                          : p.global_blocks_ops[pir_op_idx_];

      PADDLE_ENFORCE_EQ(
          op->attribute("values").isa<::pir::ArrayAttribute>(),
          true,
          ::common::errors::InvalidArgument(
              "The type of attribute 'values' in %s op is not ArrayAttribute.",
              op->name()));
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
          auto res = &int_values_;
          for (size_t i = 0; i < array_list.size(); ++i) {
            res->push_back(
                array_list[i].dyn_cast<::pir::Int32Attribute>().data());
          }
        } else if (array_list[0].isa<::pir::Int64Attribute>()) {
          auto res = &int_values_;
          for (size_t i = 0; i < array_list.size(); ++i) {
            res->push_back(
                array_list[i].dyn_cast<::pir::Int64Attribute>().data());
          }
        }
      }
    }
  }

  int32_t GetMinOpsetVersion(bool verbose) override;
  void Opset17() override;

 private:
  std::vector<int64_t> axes_;
  std::vector<int64_t> starts_;
  std::vector<int64_t> ends_;
  std::vector<int64_t> steps_;
  std::vector<int64_t> shape_;
  std::vector<int64_t> decrease_axes_;
  std::vector<int64_t> none_axes_;
  std::vector<int64_t> int_values_;
  std::vector<float> fp32_values_;
  std::vector<double> fp64_values_;
};

}  // namespace paddle2onnx
