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
#include "paddle2onnx/converter.h"
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
    P2OLogger(verbose) << "Start to parse PaddlePaddle model..." << std::endl;
    P2OLogger(verbose) << "Model file path: " << model_filename << std::endl;
    P2OLogger(verbose) << "Paramters file path: " << params_filename << std::endl;
    char* out = nullptr;
    int size = 0;

    if (!Export(model_filename.c_str(), params_filename.c_str(), &out, size,
                opset_version, auto_upgrade_opset, verbose, enable_onnx_checker,
                enable_experimental_op, enable_optimize)) {
      P2OLogger(verbose) << "Paddle model convert failed." << std::endl;
      return pybind11::bytes("");
    }
    std::string onnx_proto(out, out + size);
    delete out;
    out = nullptr;
    return pybind11::bytes(onnx_proto);
  });
  m.def(
      "optimize",
      [](const std::string& model_path, const std::string& optimized_model_path,
         const std::map<std::string, std::vector<int>>& shape_infos) {
        ONNX_NAMESPACE::optimization::OptimizePaddle2ONNX(
            model_path, optimized_model_path, shape_infos);
      });

}
}  // namespace paddle2onnx
