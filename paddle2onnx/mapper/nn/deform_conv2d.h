// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

namespace paddle2onnx
{

  class DeformConv2dMapper : public Mapper
  {
  public:
    DeformConv2dMapper(const PaddleParser &p, OnnxHelper *helper, int64_t block_id,
                       int64_t op_id)
        : Mapper(p, helper, block_id, op_id)
    {
      GetAttr("deformable_groups", &deformable_groups_);
      GetAttr("strides", &strides_);
      GetAttr("paddings", &paddings_);
      GetAttr("dilations", &dilations_);
      GetAttr("groups", &groups_);
      GetAttr("im2col_step", &im2col_step_);
    }

    int32_t GetMinOpset(bool verbose) override;
    void Opset19() override;

  private:
    std::vector<int64_t> strides_;
    std::vector<int64_t> paddings_;
    std::vector<int64_t> dilations_;
    int64_t deformable_groups_;
    int64_t groups_;
    int64_t im2col_step_;
  };

} // namespace paddle2onnx
