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
#include <map>
#include <string>
#include <vector>
#include "paddle2onnx/mapper/mapper.h"

namespace paddle2onnx {

class ActivationMapper : public Mapper {
 public:
  ActivationMapper(const PaddleParser& p, int64_t block_id, int64_t op_id)
      : Mapper(p, block_id, op_id) {
    op_mapper_["relu"] = "Relu";
    op_mapper_["tanh"] = "Tanh";
    op_mapper_["log"] = "Log";
    op_mapper_["sigmoid"] = "Sigmoid";
    op_mapper_["sqrt"] = "Sqrt";
    op_mapper_["softplus"] = "Softplus";
  }

  int32_t GetMinOpset(bool verbose = false);
  void Opset7(OnnxHelper* helper);

 private:
  std::map<std::string, std::string> op_mapper_;
};

class Relu6Mapper : public Mapper {
 public:
  Relu6Mapper(const PaddleParser& p, int64_t block_id, int64_t op_id)
      : Mapper(p, block_id, op_id) {
    auto op = parser_->GetOpDesc(block_idx_, op_idx_);
    parser_->GetOpAttr(op, "threshold", &threshold_);
  }

  void Opset7(OnnxHelper* helper);

 private:
  float threshold_;
};

class PReluMapper : public Mapper {
 public:
  PReluMapper(const PaddleParser& p, int64_t block_id, int64_t op_id)
      : Mapper(p, block_id, op_id) {}

  void Opset7(OnnxHelper* helper);
};

class SeluMapper : public Mapper {
 public:
  SeluMapper(const PaddleParser& p, int64_t block_id, int64_t op_id)
      : Mapper(p, block_id, op_id) {
    auto op = parser_->GetOpDesc(block_idx_, op_idx_);
    parser_->GetOpAttr(op, "alpha", &alpha_);
    parser_->GetOpAttr(op, "scale", &scale_);
  }

  void Opset7(OnnxHelper* helper);

 private:
  float alpha_;
  float scale_;
};

class HardSigmoidMapper : public Mapper {
 public:
  HardSigmoidMapper(const PaddleParser& p, int64_t block_id, int64_t op_id)
      : Mapper(p, block_id, op_id) {
    auto op = parser_->GetOpDesc(block_idx_, op_idx_);
    parser_->GetOpAttr(op, "slope", &alpha_);
    parser_->GetOpAttr(op, "offset", &beta_);
  }

  void Opset7(OnnxHelper* helper);

 private:
  float alpha_;
  float beta_;
};

class SwishMapper : public Mapper {
 public:
  SwishMapper(const PaddleParser& p, int64_t block_id, int64_t op_id)
      : Mapper(p, block_id, op_id) {
    auto op = parser_->GetOpDesc(block_idx_, op_idx_);
    parser_->GetOpAttr(op, "beta", &beta_);
  }

  void Opset7(OnnxHelper* helper);

 private:
  float beta_;
};

class HardSwishMapper : public Mapper {
 public:
  HardSwishMapper(const PaddleParser& p, int64_t block_id, int64_t op_id)
      : Mapper(p, block_id, op_id) {
    auto op = parser_->GetOpDesc(block_idx_, op_idx_);
    parser_->GetOpAttr(op, "scale", &scale_);
    parser_->GetOpAttr(op, "offset", &offset_);
    parser_->GetOpAttr(op, "threshold", &threshold_);
  }

  void Opset7(OnnxHelper* helper);

 private:
  float scale_;
  float offset_;
  float threshold_;
};

class LeakyReluMapper : public Mapper {
 public:
  LeakyReluMapper(const PaddleParser& p, int64_t block_id, int64_t op_id)
      : Mapper(p, block_id, op_id) {
    auto op = parser_->GetOpDesc(block_idx_, op_idx_);
    parser_->GetOpAttr(op, "alpha", &alpha_);
  }

  void Opset7(OnnxHelper* helper);

 private:
  float alpha_;
};

class GeluMapper : public Mapper {
 public:
  GeluMapper(const PaddleParser& p, int64_t block_id, int64_t op_id)
      : Mapper(p, block_id, op_id) {}

  int32_t GetMinOpset(bool verbose = false) { return 9; }

  void Opset9(OnnxHelper* helper);
};
}  // namespace paddle2onnx
