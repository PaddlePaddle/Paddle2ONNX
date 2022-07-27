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
                                   bool enable_optimize, CustomOp* ops,
                                   int op_count, const char* deploy_backend) {
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

  // Add custom operator information
  if (ops != nullptr && op_count > 0) {
    for (int i = 0; i < op_count; ++i) {
      std::string op_name(ops[i].op_name, strlen(ops[i].op_name));
      std::string export_op_name(ops[i].export_op_name,
                                 strlen(ops[i].export_op_name));
      if (export_op_name == "paddle2onnx_null") {
        export_op_name = op_name;
      }
      me.custom_ops[op_name] = export_op_name;
    }
  }

  if (me.GetMinOpset(parser, false) < 0) {
    return false;
  }
  std::string onnx_model = me.Run(
      parser, opset_version, auto_upgrade_opset, verbose, enable_onnx_checker,
      enable_experimental_op, enable_optimize, deploy_backend);
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
                                   bool enable_optimize, CustomOp* ops,
                                   int op_count, const char* deploy_backend) {
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

  // Add custom operator information
  if (ops != nullptr && op_count > 0) {
    for (int i = 0; i < op_count; ++i) {
      std::string op_name(ops[i].op_name, strlen(ops[i].op_name));
      std::string export_op_name(ops[i].export_op_name,
                                 strlen(ops[i].export_op_name));
      if (export_op_name == "paddle2onnx_null") {
        export_op_name = op_name;
      }
      me.custom_ops[op_name] = export_op_name;
    }
  }

  if (me.GetMinOpset(parser, false) < 0) {
    return false;
  }
  std::string onnx_model = me.Run(
      parser, opset_version, auto_upgrade_opset, verbose, enable_onnx_checker,
      enable_experimental_op, enable_optimize, deploy_backend);
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
                             bool enable_experimental_op, bool enable_optimize,
                             CustomOp* ops, int op_count,
                             const char* deploy_backend) {
  auto parser = PaddleParser();
  P2OLogger(verbose) << "Start to parsing Paddle model..." << std::endl;
  if (!parser.Init(model, params)) {
    P2OLogger(verbose) << "Paddle model parsing failed." << std::endl;
    return false;
  }
  paddle2onnx::ModelExporter me;

  // Add custom operator information
  if (ops != nullptr && op_count > 0) {
    for (int i = 0; i < op_count; ++i) {
      std::string op_name(ops[i].op_name, strlen(ops[i].op_name));
      std::string export_op_name(ops[i].export_op_name,
                                 strlen(ops[i].export_op_name));
      if (export_op_name == "paddle2onnx_null") {
        export_op_name = op_name;
      }
      me.custom_ops[op_name] = export_op_name;
    }
  }

  std::string result = me.Run(
      parser, opset_version, auto_upgrade_opset, verbose, enable_onnx_checker,
      enable_experimental_op, enable_optimize, deploy_backend);
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
                             bool enable_experimental_op, bool enable_optimize,
                             CustomOp* ops, int op_count,
                             const char* deploy_backend) {
  auto parser = PaddleParser();
  P2OLogger(verbose) << "Start to parsing Paddle model..." << std::endl;
  if (!parser.Init(model_buffer, model_size, params_buffer, params_size)) {
    P2OLogger(verbose) << "Paddle model parsing failed." << std::endl;
    return false;
  }
  paddle2onnx::ModelExporter me;

  // Add custom operator information
  if (ops != nullptr && op_count > 0) {
    for (int i = 0; i < op_count; ++i) {
      std::string op_name(ops[i].op_name, strlen(ops[i].op_name));
      std::string export_op_name(ops[i].export_op_name,
                                 strlen(ops[i].export_op_name));
      if (export_op_name == "paddle2onnx_null") {
        export_op_name = op_name;
      }
      me.custom_ops[op_name] = export_op_name;
    }
  }

  std::string result = me.Run(
      parser, opset_version, auto_upgrade_opset, verbose, enable_onnx_checker,
      enable_experimental_op, enable_optimize, deploy_backend);
  if (result.empty()) {
    P2OLogger(verbose) << "The exported ONNX model is invalid!" << std::endl;
    return false;
  }
  *out_size = result.size();
  *out = new char[*out_size]();
  memcpy(*out, result.data(), *out_size);
  return true;
}

ModelTensorInfo::~ModelTensorInfo() {
  if (shape != nullptr) {
    delete[] shape;
    shape = nullptr;
    rank = 0;
  }
}
}  // namespace paddle2onnx
