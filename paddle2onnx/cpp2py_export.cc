#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "paddle2onnx/mapper/exporter.hpp"

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
}
}  // namespace paddle2onnx
