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

#include "paddle2onnx/converter.h"
#include <fstream>
#include <iostream>
#include <set>
#include "paddle2onnx/mapper/exporter.h"

namespace paddle2onnx {

PADDLE2ONNX_DECL bool IsExportable(
    const std::string& model, const std::string& params,
    bool from_memory_buffer, int32_t opset_version, bool auto_upgrade_opset,
    bool verbose, bool enable_onnx_checker, bool enable_experimental_op,
    bool enable_optimize) {
  auto parser = PaddleParser();
  if (!parser.Init(model, params, from_memory_buffer)) {
    return false;
  }
  paddle2onnx::ModelExporter me;
  std::set<std::string> unsupported_ops;
  if (!me.CheckIfOpSupported(parser, &unsupported_ops,
                             enable_experimental_op)) {
    return false;
  }
  if (me.GetMinOpset(parser, false) < 0) {
    return false;
  }
  me.Run(parser, opset_version, auto_upgrade_opset, verbose,
         enable_onnx_checker, enable_experimental_op, enable_optimize);
  return true;
}

PADDLE2ONNX_DECL bool Export(const std::string& model,
                             const std::string& params, std::string* out,
                             bool from_memory_buffer, int32_t opset_version,
                             bool auto_upgrade_opset, bool verbose,
                             bool enable_onnx_checker,
                             bool enable_experimental_op,
                             bool enable_optimize) {
  auto parser = PaddleParser();
  if (verbose) {
    std::cerr << "Start to parsing Paddle model..." << std::endl;
  }
  if (!parser.Init(model, params, from_memory_buffer)) {
    if (verbose) {
      std::cerr << "Paddle model parsing failed." << std::endl;
    }
    return false;
  }
  paddle2onnx::ModelExporter me;
  std::set<std::string> unsupported_ops;
  if (!me.CheckIfOpSupported(parser, &unsupported_ops,
                             enable_experimental_op)) {
    return false;
  }
  if (me.GetMinOpset(parser, false) < 0) {
    return false;
  }
  *out = me.Run(parser, opset_version, auto_upgrade_opset, verbose,
                enable_onnx_checker, enable_experimental_op, enable_optimize);
  return true;
}
}  // namespace paddle2onnx
