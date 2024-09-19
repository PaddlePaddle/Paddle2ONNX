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
class ORTQuantizeProcessor : public BaseQuantizeProcessor {
 public:
  ORTQuantizeProcessor() = default;
  virtual ~ORTQuantizeProcessor() = default;

  // According to:
  // https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/core/optimizer/qdq_transformer/selectors_actions/qdq_selector_action_transformer.cc
  void AddQDQ() override;

  void ProcessQuantizeModel(
      std::vector<std::shared_ptr<ONNX_NAMESPACE::NodeProto>> *parameters,
      std::vector<std::shared_ptr<ONNX_NAMESPACE::ValueInfoProto>> *inputs,
      std::vector<std::shared_ptr<ONNX_NAMESPACE::ValueInfoProto>> *outputs,
      std::vector<std::shared_ptr<ONNX_NAMESPACE::NodeProto>> *nodes,
      OnnxHelper *helper, const std::string &deploy_backend,
      const PaddleParser &parser,
      std::string *calibration_cache = nullptr) override;
};
}  // namespace paddle2onnx