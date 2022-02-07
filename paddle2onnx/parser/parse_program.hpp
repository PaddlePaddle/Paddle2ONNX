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
#include <fstream>
#include <string>
#include "paddle2onnx/proto/framework.pb.h"
#include "paddle2onnx/utils/utils.hpp"

namespace paddle2onnx {

bool ReadBinaryFile(const std::string& path, std::string* contents) {
  std::ifstream fin(path, std::ios::in | std::ios::binary);
  if (!fin.is_open()) {
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

std::shared_ptr<paddle2onnx::framework::proto::ProgramDesc> LoadProgram(
    const std::string& path) {
  std::string contents;
  Assert(ReadBinaryFile(path, &contents),
         "Fail to read model file " + path +
             " please make sure your model file or file path is valid.");
  auto program = std::make_shared<paddle2onnx::framework::proto::ProgramDesc>();
  program->ParseFromString(contents);
  return program;
}
}  // namespace paddle2onnx
