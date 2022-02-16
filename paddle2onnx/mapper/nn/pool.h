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

class Pool2dMapper : public Mapper {
 public:
  Pool2dMapper(const PaddleParser& p, int64_t block_id, int64_t op_id)
      : Mapper(p, block_id, op_id) {
    auto op = parser_->GetOpDesc(block_idx_, op_idx_);
    op_mapper_["max"] = {"MaxPool", "GlobalMaxPool"};
    op_mapper_["avg"] = {"AveragePool", "GlobalAveragePool"};
    parser_->GetOpAttr(op, "data_format", &data_format_);
  }
  bool IsSameSpan(const int64_t& in_size, const int64_t& out_size);
  int32_t GetMinOpset(bool verbose = false);
  void Opset7(OnnxHelper* helper);

 private:
  std::string data_format_;
  std::map<std::string, std::vector<std::string>> op_mapper_;
};

}  // namespace paddle2onnx
