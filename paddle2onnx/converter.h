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
#include <stdint.h>

#if defined(_WIN32)
#ifdef PADDLE2ONNX_LIB
#define PADDLE2ONNX_DECL __declspec(dllexport)
#else
#define PADDLE2ONNX_DECL __declspec(dllimport)
#endif  // PADDLE2ONNX_LIB
#else
#define PADDLE2ONNX_DECL __attribute__((visibility("default")))
#endif  // _WIN32

namespace paddle2onnx {

struct PADDLE2ONNX_DECL CustomOp {
  char op_name[100] = "null";
  // if export_op_name set as "paddle2onnx_null"
  // it will automaticly change to `op_name`
  char export_op_name[100] = "paddle2onnx_null";
};

PADDLE2ONNX_DECL bool IsExportable(
    const char* model, const char* params, int32_t opset_version = 11,
    bool auto_upgrade_opset = true, bool verbose = false,
    bool enable_onnx_checker = true, bool enable_experimental_op = false,
    bool enable_optimize = true, CustomOp* ops = nullptr, int op_count = 0,
    const char* deploy_backend = "onnxruntime");

PADDLE2ONNX_DECL bool IsExportable(
    const void* model_buffer, int model_size, const void* params_buffer,
    int params_size, int32_t opset_version = 11, bool auto_upgrade_opset = true,
    bool verbose = false, bool enable_onnx_checker = true,
    bool enable_experimental_op = false, bool enable_optimize = true,
    CustomOp* ops = nullptr, int op_count = 0,
    const char* deploy_backend = "onnxruntime");

PADDLE2ONNX_DECL bool Export(const char* model, const char* params, char** out,
                             int* out_size, int32_t opset_version = 11,
                             bool auto_upgrade_opset = true,
                             bool verbose = false,
                             bool enable_onnx_checker = true,
                             bool enable_experimental_op = false,
                             bool enable_optimize = true,
                             CustomOp* ops = nullptr, int op_count = 0,
                             const char* deploy_backend = "onnxruntime");

PADDLE2ONNX_DECL bool Export(
    const void* model_buffer, int model_size, const void* params_buffer,
    int params_size, char** out, int* out_size, int32_t opset_version = 11,
    bool auto_upgrade_opset = true, bool verbose = false,
    bool enable_onnx_checker = true, bool enable_experimental_op = false,
    bool enable_optimize = true, CustomOp* ops = nullptr, int op_count = 0,
    const char* deploy_backend = "onnxruntime");

struct PADDLE2ONNX_DECL ModelTensorInfo {
  char name[100];
  int32_t* shape = nullptr;
  int32_t rank = 0;
  ~ModelTensorInfo();
};

struct PADDLE2ONNX_DECL OnnxReader {
  OnnxReader(const char* model_buffer, int buffer_size);
  int NumInputs() const;
  int NumOutputs() const;
  int GetInputIndex(const char* name) const;
  int GetOutputIndex(const char* name) const;
  void GetInputInfo(int index, ModelTensorInfo* info) const;
  void GetOutputInfo(int index, ModelTensorInfo* info) const;
  // suppose the maximum number of inputs/outputs is 100
  // suppose the longest string of inputs/outputs is 200
  // suppose the biggest rank will be less than 10
  char input_names[100][200] = {""};
  char output_names[100][200] = {""};
  int32_t input_shapes[100][10];
  int32_t output_shapes[100][10];
  int32_t input_ranks[100];
  int32_t output_ranks[100];
  int num_inputs;
  int num_outputs;
};

PADDLE2ONNX_DECL bool RemoveMultiClassNMS(const char* onnx_model,
                                          int model_size, char** out_model,
                                          int* out_model_size);

struct PADDLE2ONNX_DECL PaddleReader {
  PaddleReader(const char* model_buffer, int buffer_size);
  int NumInputs() const;
  int NumOutputs() const;
  int GetInputIndex(const char* name) const;
  int GetOutputIndex(const char* name) const;
  // suppose the maximum number of inputs/outputs is 100
  // suppose the longest string of inputs/outputs is 200
  char input_names[100][200] = {""};
  char output_names[100][200] = {""};
  int num_inputs;
  int num_outputs;
  bool has_nms = false;
};

}  // namespace paddle2onnx
