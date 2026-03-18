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

class PutAlongAxisMapper : public Mapper {
 public:
  PutAlongAxisMapper(const PaddlePirParser& p,
                     OnnxHelper* helper,
                     int64_t op_id,
                     bool if_in_cf_block)
      : Mapper(p, helper, op_id, if_in_cf_block) {
    GetAttr("axis", &axis_);
    GetAttr("reduce", &reduce_);
    GetAttr("include_self", &include_self_);
  }

  int32_t GetMinOpsetVersion(bool verbose) override;
  void Opset11() override;
  void Opset16() override;
  void Opset18() override;

 private:
  int64_t axis_;
  std::string reduce_;
  bool include_self_;
};
}  // namespace paddle2onnx
