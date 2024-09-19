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

#include "paddle2onnx/mapper/quantize/other_quantize_processor.h"

namespace paddle2onnx {
void OtherQuantizeProcessor::ProcessQuantizeModel(
    std::vector<std::shared_ptr<ONNX_NAMESPACE::NodeProto>>* parameters,
    std::vector<std::shared_ptr<ONNX_NAMESPACE::ValueInfoProto>>* inputs,
    std::vector<std::shared_ptr<ONNX_NAMESPACE::ValueInfoProto>>* outputs,
    std::vector<std::shared_ptr<ONNX_NAMESPACE::NodeProto>>* nodes,
    OnnxHelper* helper, const PaddleParser& parser,
    std::string* calibration_cache) {
  BaseQuantizeProcessor::ProcessQuantizeModel(
      parameters, inputs, outputs, nodes, helper, parser, calibration_cache);

  // If deploy_backend is others, the quantization model is exported as a
  // float model + quantization table.
  RemoveAllQuantizeOps();
  std::ofstream outfile;
  outfile.open("max_range.txt", std::ios::out);
  if (!outfile.is_open()) {
    P2OLogger() << "[WARNING] Quantize model processer failed to write range "
                   "information in current location."
                << std::endl;
    return;
  }
  for (auto iter = helper_->quantize_info.begin();
       iter != helper_->quantize_info.end(); iter++) {
    std::string log = iter->first;
    auto scale = iter->second.scale_;
    if (scale.size() == 1) {
      log = log + ": " + std::to_string(scale[0] * 127);
      outfile << log << std::endl;
    }
  }
  outfile.close();
}
}  // namespace paddle2onnx