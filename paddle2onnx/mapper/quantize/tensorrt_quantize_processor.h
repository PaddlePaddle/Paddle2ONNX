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

#include "paddle2onnx/mapper/quantize/base_quantize_processor.h"

namespace paddle2onnx {
class TensorRTQuantizeProcessor : public BaseQuantizeProcessor {
 public:
  TensorRTQuantizeProcessor() = default;
  virtual ~TensorRTQuantizeProcessor() = default;

  void ProcessQuantizeModel(
      std::vector<std::shared_ptr<ONNX_NAMESPACE::NodeProto>> *parameters,
      std::vector<std::shared_ptr<ONNX_NAMESPACE::ValueInfoProto>> *inputs,
      std::vector<std::shared_ptr<ONNX_NAMESPACE::ValueInfoProto>> *outputs,
      std::vector<std::shared_ptr<ONNX_NAMESPACE::NodeProto>> *nodes,
      OnnxHelper *helper, const PaddleParser &parser,
      std::string *calibration_cache = nullptr) override;

 protected:
  // According to:
  // https://github.com/NVIDIA/TensorRT/tree/main/tools/pytorch-quantization/pytorch_quantization/nn/modules
  void AddQDQ() override;

 private:
  // Generate cache file for TensorRT8.X int8 deploy
  void GenerateCache(std::string *calibration_cache);
};
}  // namespace paddle2onnx