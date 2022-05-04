#include "deploykit/backends/ort/ort_backend.h"
#include "deploykit/backends/tensorrt/trt_backend.h"

int main() {
  auto backend = deploykit::OrtBackend();
  auto option = deploykit::OrtBackendOption();
  option.use_gpu = true;

  //  auto backend = deploykit::TrtBackend();
  //  auto option = deploykit::TrtBackendOption();
  //  option.min_shape["inputs"] = {1, 3, 224, 224};
  //  option.opt_shape["inputs"] = {4, 3, 224, 224};
  //  option.max_shape["inputs"] = {8, 3, 224, 224};

  if (!backend.InitFromPaddle("resnet50/inference.pdmodel",
                              "resnet50/inference.pdiparams", option)) {
    std::cerr << "Init failed." << std::endl;
    return -1;
  }
  std::vector<deploykit::DataBlob> inputs(1);
  inputs[0].name = "inputs";
  inputs[0].Resize({1, 3, 224, 224}, deploykit::PaddleDataType::FP32);
  std::vector<deploykit::DataBlob> outputs;
  if (!backend.Infer(inputs, &outputs)) {
    std::cerr << "Inference failed." << std::endl;
    return -1;
  }

  std::cout << "Number of outputs: " << outputs.size() << std::endl;
  for (size_t i = 0; i < outputs.size(); ++i) {
    std::cout << "Shape of outputs " << i << ": ";
    for (size_t k = 0; k < outputs[i].shape.size(); ++k) {
      std::cout << outputs[i].shape[k] << " ";
    }
    std::cout << std::endl;
  }
  return 0;
}
