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
#include <cmath>
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
    op_mapper_["sqrt"] = "Sqrt";
    op_mapper_["softplus"] = "Softplus";
    op_mapper_["exp"] = "Exp";
    op_mapper_["floor"] = "Floor";
    op_mapper_["cos"] = "Cos";
    op_mapper_["sin"] = "Sin";
    op_mapper_["round"] = "Round";
    op_mapper_["abs"] = "Abs";
    op_mapper_["acos"] = "Acos";
    op_mapper_["asin"] = "Asin";
    op_mapper_["atan"] = "Atan";
    op_mapper_["tan"] = "Tan";
    op_mapper_["ceil"] = "Ceil";
    op_mapper_["erf"] = "Erf";
    op_mapper_["softsign"] = "Softsign";
    op_mapper_["reciprocal"] = "Reciprocal";
  }

  int32_t GetMinOpsetVersion(bool verbose) override;
  void Opset7() override;

 private:
  std::map<std::string, std::string> op_mapper_;
};

class PReluMapper : public Mapper {
 public:
  PReluMapper(const PaddleParser& p, OnnxHelper* helper, int64_t block_id,
              int64_t op_id)
      : Mapper(p, helper, block_id, op_id) {}

  int32_t GetMinOpsetVersion(bool verbose) override;
  void Opset7() override;
};

class SeluMapper : public Mapper {
 public:
  SeluMapper(const PaddleParser& p, OnnxHelper* helper, int64_t block_id,
             int64_t op_id)
      : Mapper(p, helper, block_id, op_id) {
    GetAttr("alpha", &alpha_);
    GetAttr("scale", &scale_);
  }

  void Opset7() override;

 private:
  float alpha_;
  float scale_;
};

class LeakyReluMapper : public Mapper {
 public:
  LeakyReluMapper(const PaddleParser& p, OnnxHelper* helper, int64_t block_id,
                  int64_t op_id)
      : Mapper(p, helper, block_id, op_id) {
    GetAttr("alpha", &alpha_);
  }

  void Opset7() override;

 private:
  float alpha_;
};

class GeluMapper : public Mapper {
 public:
  GeluMapper(const PaddleParser& p, OnnxHelper* helper, int64_t block_id,
             int64_t op_id)
      : Mapper(p, helper, block_id, op_id) {}

  int32_t GetMinOpsetVersion(bool verbose) override {
    Logger(verbose, 9) << RequireOpset(9) << std::endl;
    return 9;
  }

  void Opset9() override;
};

class SoftMaxMapper : public Mapper {
 public:
  SoftMaxMapper(const PaddleParser& p, OnnxHelper* helper, int64_t block_id,
                int64_t op_id)
      : Mapper(p, helper, block_id, op_id) {
    if (HasAttr("axis")) {
      GetAttr("axis", &axis_);
    } else {
      axis_ = -1;
    }
  }

  void Opset7() override;
  void Opset13() override;

 private:
  int64_t axis_ = -1;
};

class BReluMapper : public Mapper {
 public:
  BReluMapper(const PaddleParser& p, OnnxHelper* helper, int64_t block_id,
              int64_t op_id)
      : Mapper(p, helper, block_id, op_id) {
    GetAttr("t_min", &t_min_);
    GetAttr("t_max", &t_max_);
  }

  void Opset7() override;

 private:
  float t_min_;
  float t_max_;
};

class EluMapper : public Mapper {
 public:
  EluMapper(const PaddleParser& p, OnnxHelper* helper, int64_t block_id,
            int64_t op_id)
      : Mapper(p, helper, block_id, op_id) {
    GetAttr("alpha", &alpha_);
  }
  void Opset7() override;

 private:
  float alpha_;
};

class MishMapper : public Mapper {
 public:
  MishMapper(const PaddleParser& p, OnnxHelper* helper, int64_t block_id,
             int64_t op_id)
      : Mapper(p, helper, block_id, op_id) {
    GetAttr("threshold", &threshold_);
  }
  int32_t GetMinOpsetVersion(bool verbose) override;
  void Opset7() override;

 private:
  float threshold_;
};

class SquareMapper : public Mapper {
 public:
  SquareMapper(const PaddleParser& p, OnnxHelper* helper, int64_t block_id,
               int64_t op_id)
      : Mapper(p, helper, block_id, op_id) {}
  void Opset7() override;
};

class SizeMapper : public Mapper {
 public:
  SizeMapper(const PaddleParser& p, OnnxHelper* helper, int64_t block_id,
             int64_t op_id)
      : Mapper(p, helper, block_id, op_id) {}
  void Opset7() override;
};

class LogSigmoidMapper : public Mapper {
 public:
  LogSigmoidMapper(const PaddleParser& p, OnnxHelper* helper, int64_t block_id,
                   int64_t op_id)
      : Mapper(p, helper, block_id, op_id) {}
  void Opset7() override;
};

class RsqrtMapper : public Mapper {
 public:
  RsqrtMapper(const PaddleParser& p, OnnxHelper* helper, int64_t block_id,
              int64_t op_id)
      : Mapper(p, helper, block_id, op_id) {}
  void Opset7() override;
};

class LogSoftmaxMapper : public Mapper {
 public:
  LogSoftmaxMapper(const PaddleParser& p, OnnxHelper* helper, int64_t block_id,
                   int64_t op_id)
      : Mapper(p, helper, block_id, op_id) {
    GetAttr("axis", &axis_);
  }
  void Opset7() override;

 private:
  int64_t axis_;
};

class SoftShrinkMapper : public Mapper {
 public:
  SoftShrinkMapper(const PaddleParser& p, OnnxHelper* helper, int64_t block_id,
                   int64_t op_id)
      : Mapper(p, helper, block_id, op_id) {
    GetAttr("lambda", &lambda_);
  }
  int32_t GetMinOpsetVersion(bool verbose) override {
    Logger(verbose, 9) << RequireOpset(9) << std::endl;
    return 9;
  }
  void Opset9() override;

 private:
  float lambda_;
};

class ThresholdedReluMapper : public Mapper {
 public:
  ThresholdedReluMapper(const PaddleParser& p, OnnxHelper* helper,
                        int64_t block_id, int64_t op_id)
      : Mapper(p, helper, block_id, op_id) {
    GetAttr("threshold", &threshold_);
  }
  int32_t GetMinOpsetVersion(bool verbose) override {
    Logger(verbose, 10) << RequireOpset(10) << std::endl;
    return 10;
  }
  void Opset10() override;

 private:
  float threshold_;
};

class TanhShrinkMapper : public Mapper {
 public:
  TanhShrinkMapper(const PaddleParser& p, OnnxHelper* helper, int64_t block_id,
                   int64_t op_id)
      : Mapper(p, helper, block_id, op_id) {}
  void Opset7() override;
};

class Log1PMapper : public Mapper {
 public:
  Log1PMapper(const PaddleParser& p, OnnxHelper* helper, int64_t block_id,
              int64_t op_id)
      : Mapper(p, helper, block_id, op_id) {}
  void Opset7() override;
};

class Log2Mapper : public Mapper {
 public:
  Log2Mapper(const PaddleParser& p, OnnxHelper* helper, int64_t block_id,
             int64_t op_id)
      : Mapper(p, helper, block_id, op_id) {}
  void Opset7() override;
};

class Log10Mapper : public Mapper {
 public:
  Log10Mapper(const PaddleParser& p, OnnxHelper* helper, int64_t block_id,
              int64_t op_id)
      : Mapper(p, helper, block_id, op_id) {}
  void Opset7() override;
};

class SiluMapper : public Mapper {
 public:
  SiluMapper(const PaddleParser& p, OnnxHelper* helper, int64_t block_id,
             int64_t op_id)
      : Mapper(p, helper, block_id, op_id) {}
  void Opset7() override;
};

}  // namespace paddle2onnx
