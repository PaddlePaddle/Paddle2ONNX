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
  Mapper(const PaddleParser& p, int32_t block_id, int32_t op_id) : parser_(&p) {
    block_idx_ = block_id;
    op_idx_ = op_id;
  }

  // the return value in [7, 15], represent the minimum opset_version
  // if return value < 0, means the op is not supported.
  virtual int32_t GetMinOpset(bool verbose) = 0;

  void Run(OnnxHelper* helper, int32_t opset_version = 7) {
    export_opset_version_ = opset_version;
    Assert(opset_version >= 7 && opset_version <= 15,
           "Paddle2ONNX only support opset_version in range of [7, 15].");
    if (opset_version == 15) {
      Opset15(helper);
    } else if (opset_version == 14) {
      Opset14(helper);
    } else if (opset_version == 13) {
      Opset13(helper);
    } else if (opset_version == 12) {
      Opset12(helper);
    } else if (opset_version == 11) {
      Opset11(helper);
    } else if (opset_version == 10) {
      Opset10(helper);
    } else if (opset_version == 9) {
      Opset9(helper);
    } else if (opset_version == 8) {
      Opset8(helper);
    } else {
      Opset7(helper);
    }
  }

  virtual void Opset15(OnnxHelper* helper) { Opset14(helper); }

  virtual void Opset14(OnnxHelper* helper) { Opset13(helper); }

  virtual void Opset13(OnnxHelper* helper) { Opset12(helper); }

  virtual void Opset12(OnnxHelper* helper) { Opset11(helper); }

  virtual void Opset11(OnnxHelper* helper) { Opset10(helper); }

  virtual void Opset10(OnnxHelper* helper) { Opset9(helper); }

  virtual void Opset9(OnnxHelper* helper) { Opset8(helper); }

  virtual void Opset8(OnnxHelper* helper) { Opset7(helper); }

  virtual void Opset7(OnnxHelper* helper) {
    Assert(false,
           "This error shouldn't happend, please report to "
           "https://github.com/PaddlePaddle/Paddle2ONNX.git.");
  }

  virtual ~Mapper() = default;
  const PaddleParser* parser_;
  int32_t block_idx_;
  int32_t op_idx_;
  int32_t export_opset_version_;
};

}  // namespace paddle2onnx
