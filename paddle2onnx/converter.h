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

namespace paddle2onnx {

// Check if model convertable from memory buffer
// If there's no parameters, set params = ""
// If the model is read from memory instead of disk,
// set from_memory_buffer = true.
// Set enable_experimental_op will support more operators,
// also may bring some risks while converting, please double checking your
// model.
bool IsConvertable(const std::string& model, const std::string& params,
                   bool from_memory_buffer = false,
                   bool enable_experimental_op = false);

// Convert Paddle model to ONNX format
// Return converted ONNX format model which is serialized to string from
// ONNX::ModelProto model : if from_memory_buffer == false, this means path of
// model file; Otherwise, its string stream of model params : if
// from_memory_buffer == false, this means path of parameters file; Otherwise,
// its string stream of parameters opset_version : Opset version of ONNX, in
// range of [7, 15] auto_upgrade_opset : Set to true, it will auto choose a
// convertable opset verbose : If print the intermediate log
// enable_onnx_checker: If use ONNX library to validate the converted ONNX model
// enable_experimental_op: If enable experimental operators conversion, may
// bring some risks while converting, please double checking your model.
std::string Convert(const std::string& model, const std::string& params,
                    int32_t opset_version = 9, bool from_memory_buffer = false,
                    bool auto_upgrade_opset = true, bool verbose = false,
                    bool enable_onnx_checker = false,
                    bool enable_experimental_op = false);

}  // namespace paddle2onnx
