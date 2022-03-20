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

class SliceMapper : public Mapper {
 public:
  SliceMapper(const PaddleParser& p, int64_t block_id, int64_t op_id)
      : Mapper(p, block_id, op_id) {
    auto op = parser_->GetOpDesc(block_idx_, op_idx_);
    parser_->GetOpAttr(op, "axes", &axes_);
    parser_->GetOpAttr(op, "starts", &starts_);
    parser_->GetOpAttr(op, "ends", &ends_);
    if (parser_->OpHasAttr(op, "strides")) {
      parser_->GetOpAttr(op, "strides", &strides_);
    }
    if (parser_->OpHasAttr(op, "decrease_axis_")) {
      parser_->GetOpAttr(op, "decrease_axis", &decrease_axis_);
    }
  }

  int32_t GetMinOpset(bool verbose = false);
  void Opset7(OnnxHelper* helper);
  void Opset10(OnnxHelper* helper);

 private:
  std::vector<int64_t> DecreaseAxis();
  std::vector<int64_t> axes_;
  std::vector<int64_t> starts_;
  std::vector<int64_t> ends_;
  std::vector<int64_t> strides_;
  std::vector<int64_t> decrease_axis_;
};

}  // namespace paddle2onnx
