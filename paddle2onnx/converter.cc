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
#include <string>
#include "paddle2onnx/mapper/exporter.h"

namespace paddle2onnx {

PADDLE2ONNX_DECL bool IsExportable(const char* model, const char* params,
                                   int32_t opset_version,
                                   bool auto_upgrade_opset, bool verbose,
                                   bool enable_onnx_checker,
                                   bool enable_experimental_op,
                                   bool enable_optimize) {
  auto parser = PaddleParser();
  if (!parser.Init(model, params)) {
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
  std::string onnx_model =
      me.Run(parser, opset_version, auto_upgrade_opset, verbose,
             enable_onnx_checker, enable_experimental_op, enable_optimize);
  if (onnx_model.empty()) {
    P2OLogger(verbose) << "The exported ONNX model is invalid!" << std::endl;
    return false;
  }
  return true;
}

PADDLE2ONNX_DECL bool IsExportable(const void* model_buffer, int model_size,
                                   const void* params_buffer, int params_size,
                                   int32_t opset_version,
                                   bool auto_upgrade_opset, bool verbose,
                                   bool enable_onnx_checker,
                                   bool enable_experimental_op,
                                   bool enable_optimize) {
  auto parser = PaddleParser();
  if (!parser.Init(model_buffer, model_size, params_buffer, params_size)) {
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
  std::string onnx_model =
      me.Run(parser, opset_version, auto_upgrade_opset, verbose,
             enable_onnx_checker, enable_experimental_op, enable_optimize);
  if (onnx_model.empty()) {
    P2OLogger(verbose) << "The exported ONNX model is invalid!" << std::endl;
    return false;
  }
  return true;
}

PADDLE2ONNX_DECL bool Export(const char* model, const char* params, char** out,
                             int* out_size, int32_t opset_version,
                             bool auto_upgrade_opset, bool verbose,
                             bool enable_onnx_checker,
                             bool enable_experimental_op,
                             bool enable_optimize) {
  auto parser = PaddleParser();
  P2OLogger(verbose) << "Start to parsing Paddle model..." << std::endl;
  if (!parser.Init(model, params)) {
    P2OLogger(verbose) << "Paddle model parsing failed." << std::endl;
    return false;
  }
  paddle2onnx::ModelExporter me;
  std::string result =
      me.Run(parser, opset_version, auto_upgrade_opset, verbose,
             enable_onnx_checker, enable_experimental_op, enable_optimize);
  if (result.empty()) {
    P2OLogger(verbose) << "The exported ONNX model is invalid!" << std::endl;
    return false;
  }
  *out_size = result.size();
  *out = new char[*out_size]();
  memcpy(*out, result.data(), *out_size);
  return true;
}

PADDLE2ONNX_DECL bool Export(const void* model_buffer, int model_size,
                             const void* params_buffer, int params_size,
                             char** out, int* out_size, int32_t opset_version,
                             bool auto_upgrade_opset, bool verbose,
                             bool enable_onnx_checker,
                             bool enable_experimental_op,
                             bool enable_optimize) {
  auto parser = PaddleParser();
  P2OLogger(verbose) << "Start to parsing Paddle model..." << std::endl;
  if (!parser.Init(model_buffer, model_size, params_buffer, params_size)) {
    P2OLogger(verbose) << "Paddle model parsing failed." << std::endl;
    return false;
  }
  paddle2onnx::ModelExporter me;
  std::string result =
      me.Run(parser, opset_version, auto_upgrade_opset, verbose,
             enable_onnx_checker, enable_experimental_op, enable_optimize);
  if (result.empty()) {
    P2OLogger(verbose) << "The exported ONNX model is invalid!" << std::endl;
    return false;
  }
  *out_size = result.size();
  *out = new char[*out_size]();
  memcpy(*out, result.data(), *out_size);
  return true;
}


PADDLE2ONNX_DECL OnnxReader::OnnxReader(const char* model_buffer, int buffer_size) {
  ONNX_NAMESPACE::ModelProto model;
  std::string content(model_buffer, model_buffer + buffer_size);
  model.ParseFromString(content);
  num_inputs = model.graph().input_size();
  num_outputs = model.graph().output_size();
  Assert(num_inputs <= 100, "The number of inputs is exceed 100, unexpected situation."); 
  Assert(num_outputs <= 100, "The number of outputs is exceed 100, unexpected situation."); 
  for (int i = 0; i < num_inputs; ++i) {
    memcpy(input_names[i], model.graph().input(i).name().c_str(), model.graph().input(i).name().size());
  }
  for (int i = 0; i < num_outputs; ++i) {
    memcpy(output_names[i], model.graph().output(i).name().c_str(), model.graph().output(i).name().size());
  } 
}

PADDLE2ONNX_DECL int OnnxReader::NumInputs() const {
  return num_inputs;
}

PADDLE2ONNX_DECL int OnnxReader::NumOutputs() const {
  return num_outputs;
}

PADDLE2ONNX_DECL int OnnxReader::GetInputIndex(const char* name) const {
  for (int i = 0; i < num_inputs; ++i) {
    if(strcmp(name, input_names[i]) == 0) {
      return i;
    }
  }
  return -1;
}

PADDLE2ONNX_DECL int OnnxReader::GetOutputIndex(const char* name) const {
  for (int i = 0; i < num_outputs; ++i) {
    if(strcmp(name, output_names[i]) == 0) {
      return i;
    }
  }
  return -1;
}

}  // namespace paddle2onnx
