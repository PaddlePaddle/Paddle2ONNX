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
#include "paddle2onnx/mapper/data_helper.hpp"
#include "paddle2onnx/mapper/onnx_helper.hpp"
#include "paddle2onnx/mapper/register_mapper.hpp"
#include "paddle2onnx/parser/parser.hpp"

namespace paddle2onnx {

class Mapper {
 public:
  Mapper() {}
  Mapper(const PaddleParser& p, int32_t block_id, int32_t op_id) : parser(&p) {
    block_idx = block_id;
    op_idx = op_id;
  }

  // the return value in [7, 15], represent the minimum opset_version
  // if return value < 0, means the op is not supported.
  virtual int32_t GetMinOpset(bool verbose) = 0;

  void Run(std::vector<std::shared_ptr<ONNX_NAMESPACE::NodeProto>>* nodes,
           int32_t opset_version = 7) {
    export_opset_version = opset_version;
    Assert(opset_version >= 7 && opset_version <= 15,
           "Paddle2ONNX only support opset_version in range of [7, 15].");
    if (opset_version == 15) {
      Opset15(nodes);
    } else if (opset_version == 14) {
      Opset14(nodes);
    } else if (opset_version == 13) {
      Opset13(nodes);
    } else if (opset_version == 12) {
      Opset12(nodes);
    } else if (opset_version == 11) {
      Opset11(nodes);
    } else if (opset_version == 10) {
      Opset10(nodes);
    } else if (opset_version == 9) {
      Opset9(nodes);
    } else if (opset_version == 8) {
      Opset8(nodes);
    } else {
      Opset7(nodes);
    }
  }

  virtual void Opset15(
      std::vector<std::shared_ptr<ONNX_NAMESPACE::NodeProto>>* nodes) {
    Opset14(nodes);
  }

  virtual void Opset14(
      std::vector<std::shared_ptr<ONNX_NAMESPACE::NodeProto>>* nodes) {
    Opset13(nodes);
  }

  virtual void Opset13(
      std::vector<std::shared_ptr<ONNX_NAMESPACE::NodeProto>>* nodes) {
    Opset12(nodes);
  }

  virtual void Opset12(
      std::vector<std::shared_ptr<ONNX_NAMESPACE::NodeProto>>* nodes) {
    Opset11(nodes);
  }

  virtual void Opset11(
      std::vector<std::shared_ptr<ONNX_NAMESPACE::NodeProto>>* nodes) {
    Opset10(nodes);
  }

  virtual void Opset10(
      std::vector<std::shared_ptr<ONNX_NAMESPACE::NodeProto>>* nodes) {
    Opset9(nodes);
  }

  virtual void Opset9(
      std::vector<std::shared_ptr<ONNX_NAMESPACE::NodeProto>>* nodes) {
    Opset8(nodes);
  }

  virtual void Opset8(
      std::vector<std::shared_ptr<ONNX_NAMESPACE::NodeProto>>* nodes) {
    Opset7(nodes);
  }

  virtual void Opset7(
      std::vector<std::shared_ptr<ONNX_NAMESPACE::NodeProto>>* nodes) {
    Assert(false,
           "This error shouldn't happend, please report to "
           "https://github.com/PaddlePaddle/Paddle2ONNX.git.");
  }

  virtual ~Mapper() = default;
  const PaddleParser* parser;
  int32_t block_idx;
  int32_t op_idx;
  int32_t export_opset_version;
};

}  // namespace paddle2onnx
