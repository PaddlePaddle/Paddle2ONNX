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

class FullMapper : public Mapper {
 public:
  // Only for PIR
  FullMapper(const PaddlePirParser& p,
             OnnxHelper* helper,
             int64_t op_id,
             bool if_in_cf_block)
      : Mapper(p, helper, op_id, if_in_cf_block) {
    GetAttr("dtype", &dtype_);
    GetScalarAttr("value", &value_);
    GetAttr("shape", &shape_);
  }

  void Opset7() override;

 private:
  int64_t dtype_;
  ScalarData value_;
  std::vector<int64_t> shape_;
};

}  // namespace paddle2onnx
