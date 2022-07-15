#include <fstream>
#include <iostream>
#include <set>
#include <string>
#include "paddle2onnx/converter.h"
#include "paddle2onnx/mapper/exporter.h"

namespace paddle2onnx {

PADDLE2ONNX_DECL OnnxReader::OnnxReader(const char* model_buffer,
                                        int buffer_size) {
  ONNX_NAMESPACE::ModelProto model;
  std::string content(model_buffer, model_buffer + buffer_size);
  model.ParseFromString(content);
  num_inputs = model.graph().input_size();
  num_outputs = model.graph().output_size();
  Assert(num_inputs <= 100,
         "The number of inputs is exceed 100, unexpected situation.");
  Assert(num_outputs <= 100,
         "The number of outputs is exceed 100, unexpected situation.");
  for (int i = 0; i < num_inputs; ++i) {
    memcpy(input_names[i], model.graph().input(i).name().c_str(),
           model.graph().input(i).name().size());
  }
  for (int i = 0; i < num_outputs; ++i) {
    memcpy(output_names[i], model.graph().output(i).name().c_str(),
           model.graph().output(i).name().size());
  }
}

PADDLE2ONNX_DECL int OnnxReader::NumInputs() const { return num_inputs; }

PADDLE2ONNX_DECL int OnnxReader::NumOutputs() const { return num_outputs; }

PADDLE2ONNX_DECL int OnnxReader::GetInputIndex(const char* name) const {
  for (int i = 0; i < num_inputs; ++i) {
    if (strcmp(name, input_names[i]) == 0) {
      return i;
    }
  }
  return -1;
}

PADDLE2ONNX_DECL int OnnxReader::GetOutputIndex(const char* name) const {
  for (int i = 0; i < num_outputs; ++i) {
    if (strcmp(name, output_names[i]) == 0) {
      return i;
    }
  }
  return -1;
}

}  // namespace paddle2onnx
