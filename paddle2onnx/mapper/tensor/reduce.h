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

class ReduceMapper : public Mapper {
 public:
  ReduceMapper(const PaddleParser& p, OnnxHelper* helper, int64_t block_id,
               int64_t op_id, std::string name={})
      : Mapper(p, helper, block_id, op_id, name) {
    if (OpType() == "logsumexp") {
      GetAttr("keepdim", &keep_dim_);
      GetAttr("reduce_all", &reduce_all_);
      GetAttr("axis", &dim_);
    } else {
      GetAttr("keep_dim", &keep_dim_);
      GetAttr("reduce_all", &reduce_all_);
      GetAttr("in_dtype", &in_dtype_);
      GetAttr("out_dtype", &out_dtype_);
      GetAttr("dim", &dim_);
    }
  }
  void Opset7();

 private:
  bool keep_dim_;
  bool reduce_all_;
  int64_t in_dtype_;
  int64_t out_dtype_;
  std::vector<int64_t> dim_;
};

}  // namespace paddle2onnx
