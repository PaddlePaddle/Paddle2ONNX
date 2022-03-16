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
  }

  int32_t GetMinOpset(bool verbose = false);
  void Opset7(OnnxHelper* helper);
  void Opset10(OnnxHelper* helper);

 private:
  std::vector<int64_t> axes_;
  bool GetNodeAttrValue(const std::string& attr_name,
                        const std::string& attr_tensor_name,
                        const std::string& attr_tensor_list_name,
                        std::vector<int64_t>* val, std::string* val_tensor,
                        const bool& return_list = false,
                        OnnxHelper* helper = nullptr);
  std::vector<int64_t> DecreaseAxis();
};

}  // namespace paddle2onnx