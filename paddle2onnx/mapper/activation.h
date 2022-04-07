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
  ActivationMapper(const PaddleParser& p, OnnxHelper* helper, int64_t block_id,
                   int64_t op_id)
      : Mapper(p, helper, block_id, op_id) {
    op_mapper_["relu"] = "Relu";
    op_mapper_["tanh"] = "Tanh";
    op_mapper_["log"] = "Log";
    op_mapper_["sigmoid"] = "Sigmoid";
    op_mapper_["sqrt"] = "Sqrt";
    op_mapper_["softplus"] = "Softplus";
    op_mapper_["exp"] = "Exp";
    op_mapper_["floor"] = "Floor";
    op_mapper_["cos"] = "Cos";
    op_mapper_["sin"] = "Sin";
    op_mapper_["round"] = "Round";
  }

  int32_t GetMinOpset(bool verbose = false);
  void Opset7();

 private:
  std::map<std::string, std::string> op_mapper_;
};

class Relu6Mapper : public Mapper {
 public:
  Relu6Mapper(const PaddleParser& p, OnnxHelper* helper, int64_t block_id,
              int64_t op_id)
      : Mapper(p, helper, block_id, op_id) {
    GetAttr("threshold", &threshold_);
  }

  void Opset7();

 private:
  float threshold_;
};

class PReluMapper : public Mapper {
 public:
  PReluMapper(const PaddleParser& p, OnnxHelper* helper, int64_t block_id,
              int64_t op_id)
      : Mapper(p, helper, block_id, op_id) {}

  int32_t GetMinOpset(bool verbose = false);
  void Opset7();
};

class SeluMapper : public Mapper {
 public:
  SeluMapper(const PaddleParser& p, OnnxHelper* helper, int64_t block_id,
             int64_t op_id)
      : Mapper(p, helper, block_id, op_id) {
    GetAttr("alpha", &alpha_);
    GetAttr("scale", &scale_);
  }

  void Opset7();

 private:
  float alpha_;
  float scale_;
};

class HardSigmoidMapper : public Mapper {
 public:
  HardSigmoidMapper(const PaddleParser& p, OnnxHelper* helper, int64_t block_id,
                    int64_t op_id)
      : Mapper(p, helper, block_id, op_id) {
    GetAttr("slope", &alpha_);
    GetAttr("offset", &beta_);
  }

  void Opset7();

 private:
  float alpha_;
  float beta_;
};

class SwishMapper : public Mapper {
 public:
  SwishMapper(const PaddleParser& p, OnnxHelper* helper, int64_t block_id,
              int64_t op_id)
      : Mapper(p, helper, block_id, op_id) {
    GetAttr("beta", &beta_);
  }

  void Opset7();

 private:
  float beta_;
};

class HardSwishMapper : public Mapper {
 public:
  HardSwishMapper(const PaddleParser& p, OnnxHelper* helper, int64_t block_id,
                  int64_t op_id)
      : Mapper(p, helper, block_id, op_id) {
    GetAttr("scale", &scale_);
    GetAttr("offset", &offset_);
    GetAttr("threshold", &threshold_);
  }

  void Opset7();

 private:
  float scale_;
  float offset_;
  float threshold_;
};

class LeakyReluMapper : public Mapper {
 public:
  LeakyReluMapper(const PaddleParser& p, OnnxHelper* helper, int64_t block_id,
                  int64_t op_id)
      : Mapper(p, helper, block_id, op_id) {
    GetAttr("alpha", &alpha_);
  }

  void Opset7();

 private:
  float alpha_;
};

class GeluMapper : public Mapper {
 public:
  GeluMapper(const PaddleParser& p, OnnxHelper* helper, int64_t block_id,
             int64_t op_id)
      : Mapper(p, helper, block_id, op_id) {}

  int32_t GetMinOpset(bool verbose = false) {
    Logger(verbose, 9) << RequireOpset(9) << std::endl;
    return 9;
  }

  void Opset9();
};

class SoftMaxMapper : public Mapper {
 public:
  SoftMaxMapper(const PaddleParser& p, OnnxHelper* helper, int64_t block_id,
                int64_t op_id)
      : Mapper(p, helper, block_id, op_id) {
    GetAttr("axis", &axis_);
  }

  void Opset7();
  void Opset13();

 private:
  int64_t axis_ = -1;
};
}  // namespace paddle2onnx
