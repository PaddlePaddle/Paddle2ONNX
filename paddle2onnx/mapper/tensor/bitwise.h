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
#include <map>
#include "paddle2onnx/mapper/mapper.h"

namespace paddle2onnx
{

  class BitWiseMapper : public Mapper {
  public:
    BitWiseMapper(const PaddleParser &p, OnnxHelper *helper, int64_t block_id,
                  int64_t op_id)
        : Mapper(p, helper, block_id, op_id) {
      op_mapper_["bitwise_and"] = "BitwiseAnd";
      op_mapper_["bitwise_not"] = "BitwiseNot";
      op_mapper_["bitwise_or"] = "BitwiseOr";
      op_mapper_["bitwise_xor"] = "BitwiseXor";
      paddle_type_ = OpType();
      onnx_bitwise_type_ = op_mapper_.find(paddle_type_)->second;
      onnx_elemwise_type_ = onnx_bitwise_type_.substr(7);
    }
    int32_t GetMinOpsetVersion(bool verbose) override;
    void Opset7() override;
    void Opset18() override;

  private:
    std::map<std::string, std::string> op_mapper_; 
    std::string onnx_bitwise_type_;
    std::string onnx_elemwise_type_;
    std::string paddle_type_;
  };

} // namespace paddle2onnx
