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
#include "paddle2onnx/mapper/exporter.hpp"

int main(int argc, char* argv[]) {
  //  paddle2onnx::LoadProgram("ResNet18/inference.pdmodel");
  //  std::vector<paddle2onnx::Weight> weights;
  //  paddle2onnx::LoadParams("ResNet18/inference.pdiparams", &weights);
  if (argc == 1) {
    std::cerr << "Paddle2ONNX Usage(params_file_path is optional):   "
              << "    ./p2o_exec model_file_path  params_file_path"
              << std::endl;
  }
  auto parser = paddle2onnx::PaddleParser();
  if (argc == 2) {
    parser.Init(argv[1]);
  } else if (argc == 3) {
    parser.Init(argv[1], argv[2]);
  }
  paddle2onnx::ModelExporter me;
  auto onnx_proto = me.Run(parser, 7, true, true);
  std::fstream out("model.onnx", std::ios::out | std::ios::binary);
  out << onnx_proto;
  out.close();

  std::cout << "Hello world" << std::endl;
  //  std::cout << "Length of weights " << weights.size() << std::endl;
  return 0;
}
