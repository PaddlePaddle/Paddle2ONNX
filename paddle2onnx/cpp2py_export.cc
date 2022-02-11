//   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <vector>
#include "paddle2onnx/mapper/exporter.h"

namespace paddle2onnx {

PYBIND11_MODULE(paddle2onnx_cpp2py_export, m) {
  m.doc() = "Paddle2ONNX: export PaddlePaddle to ONNX";
  m.def("export", [](const std::string& model_filename,
                     const std::string& params_filename, int opset_version = 9,
                     bool auto_upgrade_opset = true, bool verbose = true) {
    auto parser = PaddleParser();
    if (params_filename != "") {
      parser.Init(model_filename, params_filename);
    } else {
      parser.Init(model_filename);
    }
    ModelExporter me;
    auto onnx_proto =
        me.Run(parser, opset_version, auto_upgrade_opset, verbose);
    return pybind11::bytes(onnx_proto);
  });
  m.def("check_op", [](const std::string& model_filename,
                       const std::string& params_filename) {
    auto parser = PaddleParser();
    if (params_filename != "") {
      parser.Init(model_filename, params_filename);
    } else {
      parser.Init(model_filename);
    }
    auto prog = parser.prog;

    std::vector<std::string> op_list;
    for (auto i = 0; i < prog->blocks_size(); ++i) {
      for (auto j = 0; j < prog->blocks(i).ops_size(); ++j) {
        if (prog->blocks(i).ops(j).type() == "feed") {
          continue;
        }
        if (prog->blocks(i).ops(j).type() == "fetch") {
          continue;
        }
        op_list.push_back(prog->blocks(i).ops(j).type());
      }
    }
    return op_list;
  });
}
}  // namespace paddle2onnx
