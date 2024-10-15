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

class BatchNormMapper : public Mapper {
 public:
  BatchNormMapper(const PaddleParser& p,
                  OnnxHelper* helper,
                  int64_t block_id,
                  int64_t op_id)
      : Mapper(p, helper, block_id, op_id) {
    GetAttr("epsilon", &epsilon_);
    GetAttr("momentum", &momentum_);
  }

  BatchNormMapper(const PaddlePirParser& p,
                  OnnxHelper* helper,
                  int64_t op_id,
                  bool c)
      : Mapper(p, helper, op_id, c) {
    GetAttr("is_test", &is_test_);
    GetAttr("use_global_stats", &use_global_stats_);
    GetAttr("trainable_statistics", &trainable_statistics_);
    GetAttr("epsilon", &epsilon_);
    GetAttr("momentum", &momentum_);
    GetAttr("data_format", &data_format_);
  }

  void Opset7() override;

 private:
  bool is_test_;
  bool use_global_stats_;
  bool trainable_statistics_;
  float epsilon_;
  float momentum_;
  std::string data_format_;
};

}  // namespace paddle2onnx
