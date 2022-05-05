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
#include "paddle2onnx/optimizer/paddle2onnx_optimizer.h"

namespace paddle2onnx {

PYBIND11_MODULE(paddle2onnx_cpp2py_export, m) {
  m.doc() = "Paddle2ONNX: export PaddlePaddle to ONNX";
  m.def("export", [](const std::string& model_filename,
                     const std::string& params_filename, int opset_version = 9,
                     bool auto_upgrade_opset = true, bool verbose = true,
                     bool enable_onnx_checker = true,
                     bool enable_experimental_op = true,
                     bool enable_optimize = true) {
    P2OLogger(verbose) << "Start to parse PaddlePaddle model(model file: "
                       << model_filename
                       << ", parameters file: " << params_filename << std::endl;
    auto parser = PaddleParser();
    if (params_filename != "") {
      parser.Init(model_filename, params_filename);
    } else {
      parser.Init(model_filename);
    }
    P2OLogger(verbose) << "Model loaded, start to converting..." << std::endl;
    ModelExporter me;
    auto onnx_proto =
        me.Run(parser, opset_version, auto_upgrade_opset, verbose,
               enable_onnx_checker, enable_experimental_op, enable_optimize);
    return pybind11::bytes(onnx_proto);
  });
  m.def(
      "optimize",
      [](const std::string& model_path, const std::string& optimized_model_path,
         const std::map<std::string, std::vector<int>>& shape_infos) {
        ONNX_NAMESPACE::optimization::OptimizePaddle2ONNX(
            model_path, optimized_model_path, shape_infos);
      });

  m.def("get_paddle_ops", [](const std::string& model_filename,
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

  // This interface can output all developed OPs and write them to the file_path
  m.def("get_all_registered_ops", [](const std::string& file_path) {
    int64_t total_ops = MapperHelper::Get()->GetAllOps(file_path);
    return total_ops;
  });
}
}  // namespace paddle2onnx
