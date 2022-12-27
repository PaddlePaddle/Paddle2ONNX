#include <fstream>
#include <iostream>
#include "paddle2onnx/converter.h"

int main(int argc, char* argv[]) {
  if (argc == 1) {
    std::cerr << "Paddle2ONNX Usage(params_file_path and onnx_model_save_path "
                 "is optional):   "
              << "    ./p2o_exec model_file_path  params_file_path  "
                 "onnx_model_save_path"
              << std::endl;
  }
  char* onnx_model;
  int out_size;
  std::string onnx_model_save_path = "model.onnx";
  if (argc == 2) {
    std::string paddle_model_path = argv[1];
    if (!paddle2onnx::Export(paddle_model_path.c_str(), "", &onnx_model,
                             &out_size)) {
      std::cerr << "Model convert failed." << std::endl;
      return -1;
    }
  } else if (argc == 3) {
    std::string paddle_model_path = argv[1];
    std::string params_file_path = argv[2];
    if (!paddle2onnx::Export(paddle_model_path.c_str(),
                             params_file_path.c_str(), &onnx_model,
                             &out_size)) {
      std::cerr << "Model converte failed." << std::endl;
      return -1;
    }
  } else if (argc == 4) {
    std::string paddle_model_path = argv[1];
    std::string params_file_path = argv[2];
    std::string onnx_model_save_path = argv[3];
    if (!paddle2onnx::Export(paddle_model_path.c_str(),
                             params_file_path.c_str(), &onnx_model,
                             &out_size)) {
      std::cerr << "Model converte failed." << std::endl;
      return -1;
    }
  }
  std::string onnx_proto(onnx_model, onnx_model + out_size);
  std::fstream out(onnx_model_save_path, std::ios::out | std::ios::binary);
  out << onnx_proto;
  out.close();

  return 0;
}
