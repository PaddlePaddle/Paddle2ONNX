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
#include <vector>

#include "paddle2onnx/mapper/mapper.h"

namespace paddle2onnx {

class SoftmaxCrossEntropyLossMapper : public Mapper {
 public:
  SoftmaxCrossEntropyLossMapper(const PaddleParser& p, OnnxHelper* helper,
                                int64_t block_id, int64_t op_id)
      : Mapper(p, helper, block_id, op_id) {
    GetAttr("axis", &axis_);
    GetAttr("soft_label", &soft_label_);
    GetAttr("ignore_index", &ignore_index_);
  }
  int32_t GetMinOpsetVersion(bool verbose) override;
  void Opset12() override;

 private:
  int64_t axis_ = -1;
  bool soft_label_ = false;
  int64_t ignore_index_ = -100;
};

}  // namespace paddle2onnx
