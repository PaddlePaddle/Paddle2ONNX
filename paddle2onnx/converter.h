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
#include <string>

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

PADDLE2ONNX_DECL bool IsExportable(
    const std::string& model, const std::string& params,
    bool from_memory_buffer = false, int32_t opset_version = 15,
    bool auto_upgrade_opset = true, bool verbose = false,
    bool enable_onnx_checker = true, bool enable_experimental_op = false,
    bool enable_optimize = false);

PADDLE2ONNX_DECL bool Export(
    const std::string& model, const std::string& params, std::string* out,
    bool from_memory_buffer = false, int32_t opset_version = 15,
    bool auto_upgrade_opset = true, bool verbose = false,
    bool enable_onnx_checker = true, bool enable_experimental_op = false,
    bool enable_optimize = false);

}  // namespace paddle2onnx
