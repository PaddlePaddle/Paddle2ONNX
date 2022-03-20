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
  AssignValueMapper(const PaddleParser& p, int64_t block_id, int64_t op_id)
      : Mapper(p, block_id, op_id) {
    auto op = parser_->GetOpDesc(block_idx_, op_idx_);
    parser_->GetOpAttr(op, "dtype", &dtype_);
    parser_->GetOpAttr(op, "shape", &shape_);
    int32_t dtype = static_cast<int32_t>(dtype_);
    if (dtype == P2ODataType::INT32) {
      parser_->GetOpAttr(op, "int32_values", &int64_values_);
    } else if (dtype == P2ODataType::FP32) {
      parser_->GetOpAttr(op, "fp32_values", &fp32_values_);
    } else if (dtype == P2ODataType::INT64) {
      parser_->GetOpAttr(op, "int64_values", &int64_values_);
    }
  }
  int32_t GetMinOpset(bool verbose = false);
  void Opset7(OnnxHelper* helper);

 private:
  std::vector<float> fp32_values_;
  std::vector<int64_t> int64_values_;
  std::vector<int64_t> shape_;
  int64_t dtype_;
};

}  // namespace paddle2onnx
