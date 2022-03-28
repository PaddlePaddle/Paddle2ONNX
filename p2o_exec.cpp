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

#include <fstream>
#include <iostream>
#include "paddle2onnx/converter.h"

bool ReadBinaryFile(const std::string& path, std::string* contents) {
  std::ifstream fin(path, std::ios::in | std::ios::binary);
  if (!fin.is_open()) {
    std::cerr << "Fail to read file " << path
              << ", please make sure your model file or file path is valid."
              << std::endl;
    return false;
  }
  fin.seekg(0, std::ios::end);
  contents->clear();
  contents->resize(fin.tellg());
  fin.seekg(0, std::ios::beg);
  fin.read(&(contents->at(0)), contents->size());
  fin.close();
  return true;
}

int main(int argc, char* argv[]) {
  if (argc == 1) {
    std::cerr << "Paddle2ONNX Usage(params_file_path is optional):   "
              << "    ./p2o_exec model_file_path  params_file_path"
              << std::endl;
  }
  std::string onnx_model;
  // PADDLE2ONNX_DECL bool Export(
  //    const std::string& model, const std::string& params, std::string* out,
  //    bool from_memory_buffer = false, int32_t opset_version = 11,
  //    bool auto_upgrade_opset = true, bool verbose = false,
  //    bool enable_onnx_checker = true, bool enable_experimental_op = false,
  //    bool enable_optimize = true);
  if (argc == 2) {
    std::string model_buffer;
    if (!ReadBinaryFile(argv[1], &model_buffer)) {
      return -1;
    }
    if (!paddle2onnx::Export(model_buffer, "", &onnx_model, true, 12, true,
                             true, false, true, false)) {
      std::cerr << "Model convert failed." << std::endl;
      return -1;
    }
  } else if (argc == 3) {
    std::string model_buffer;
    if (!ReadBinaryFile(argv[1], &model_buffer)) {
      return -1;
    }
    std::string params_buffer;
    if (!ReadBinaryFile(argv[2], &params_buffer)) {
      return -1;
    }
    if (!paddle2onnx::Export(model_buffer, params_buffer, &onnx_model, true, 7,
                             true, true, true, true, true)) {
      std::cerr << "Model converte failed." << std::endl;
      return -1;
    }
  }
  std::fstream out("model.onnx", std::ios::out | std::ios::binary);
  out << onnx_model;
  out.close();

  std::cout << "Hello world" << std::endl;
  //  std::cout << "Length of weights " << weights.size() << std::endl;
  return 0;
}
