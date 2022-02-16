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

bool IsConvertable(const std::string& model, const std::string& params,
                   bool from_memory_buffer, bool enable_experimental_op) {
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
  return true;
}

std::string Convert(const std::string& model, const std::string& params,
                    int32_t opset_version, bool from_memory_buffer,
                    bool auto_upgrade_opset, bool verbose,
                    bool enable_onnx_checker, bool enable_experimental_op) {
  auto parser = PaddleParser();
  Assert(parser.Init(model, params, from_memory_buffer),
         "Unable to parse PaddlePaddle model.");
  paddle2onnx::ModelExporter me;
  return me.Run(parser, opset_version, auto_upgrade_opset, verbose,
                enable_onnx_checker, enable_experimental_op);
}

}  // namespace paddle2onnx
