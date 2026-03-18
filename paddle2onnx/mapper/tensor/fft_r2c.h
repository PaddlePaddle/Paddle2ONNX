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

class FftR2cMapper : public Mapper {
 public:
  FftR2cMapper(const PaddlePirParser& p,
               OnnxHelper* helper,
               int64_t op_id,
               bool c)
      : Mapper(p, helper, op_id, c) {
    in_pir_mode = true;
    GetAttr("normalization", &normalization_);
    GetAttr("onesided", &onesided_);
    GetAttr("forward", &forward_);
    GetAttr("axes", &axes_);
  }

  int32_t GetMinOpsetVersion(bool verbose) override;
  void Opset17() override;

 private:
  std::string normalization_;
  bool onesided_;
  bool forward_;
  std::vector<int64_t> axes_;
};

}  // namespace paddle2onnx
