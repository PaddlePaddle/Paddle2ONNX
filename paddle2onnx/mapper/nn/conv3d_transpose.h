// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

class Conv3dTransposeMapper : public Mapper {
 public:
  Conv3dTransposeMapper(const PaddlePirParser& p,
                        OnnxHelper* helper,
                        int64_t i,
                        bool c)
      : Mapper(p, helper, i, c) {
    GetAttr("groups", &groups_);
    GetAttr("dilations", &dilations_);
    GetAttr("strides", &strides_);
    GetAttr("paddings", &paddings_);
    GetAttr("padding_algorithm", &padding_algorithm_);
    GetAttr("data_format", &data_format_);

    if (HasAttr("output_padding")) {
      GetAttr("output_padding", &output_padding_);
    }
    if (HasAttr("output_size")) {
      GetAttr("output_size", &output_size_);
    }
  }

  int32_t GetMinOpsetVersion(bool verbose) override;
  void Opset7() override;

 private:
  std::vector<int64_t> dilations_;
  std::vector<int64_t> strides_;
  std::vector<int64_t> paddings_;
  std::vector<int64_t> output_padding_;
  std::vector<int64_t> output_size_;
  std::string padding_algorithm_;
  std::string data_format_;
  int64_t groups_;
};

}  // namespace paddle2onnx
