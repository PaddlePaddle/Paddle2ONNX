#include <fstream>
#include <iostream>
#include <set>
#include <string>
#include "paddle2onnx/converter.h"
#include "paddle2onnx/mapper/exporter.h"
#include "paddle2onnx/optimizer/paddle2onnx_optimizer.h"

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

    auto& shape = model.graph().input(i).type().tensor_type().shape();
    int dim_size = shape.dim_size();
    input_ranks[i] = dim_size;
    for (int j = 0; j < dim_size; ++j) {
      input_shapes[i][j] = static_cast<int32_t>(shape.dim(j).dim_value());
    }
  }
  for (int i = 0; i < num_outputs; ++i) {
    memcpy(output_names[i], model.graph().output(i).name().c_str(),
           model.graph().output(i).name().size());

    auto& shape = model.graph().output(i).type().tensor_type().shape();
    int dim_size = shape.dim_size();
    output_ranks[i] = dim_size;
    for (int j = 0; j < dim_size; ++j) {
      output_shapes[i][j] = static_cast<int32_t>(shape.dim(j).dim_value());
    }
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

PADDLE2ONNX_DECL void OnnxReader::GetInputInfo(int index,
                                               ModelTensorInfo* info) const {
  Assert(index < NumInputs(), "The index:" + std::to_string(index) +
                                  " must be less than number of inputs:" +
                                  std::to_string(NumInputs()) + ".");
  memcpy(info->name, input_names[index], 100);
  info->rank = input_ranks[index];
  info->shape = new int32_t[info->rank];
  for (int i = 0; i < info->rank; ++i) {
    info->shape[i] = input_shapes[index][i];
  }
}

PADDLE2ONNX_DECL void OnnxReader::GetOutputInfo(int index,
                                                ModelTensorInfo* info) const {
  Assert(index < NumOutputs(), "The index:" + std::to_string(index) +
                                   " must be less than number of outputs:" +
                                   std::to_string(NumOutputs()) + ".");
  memcpy(info->name, output_names[index], 100);
  info->rank = output_ranks[index];
  info->shape = new int32_t[info->rank];
  for (int i = 0; i < info->rank; ++i) {
    info->shape[i] = output_shapes[index][i];
  }
}

PADDLE2ONNX_DECL int OnnxReader::GetOutputIndex(const char* name) const {
  for (int i = 0; i < num_outputs; ++i) {
    if (strcmp(name, output_names[i]) == 0) {
      return i;
    }
  }
  return -1;
}

bool RemoveMultiClassNMS(const char* model_buffer, int buffer_size,
                         char** out_model, int* out_model_size) {
  ONNX_NAMESPACE::ModelProto model;
  std::string content(model_buffer, model_buffer + buffer_size);
  model.ParseFromString(content);
  auto* graph = model.mutable_graph();
  int nms_index = -1;
  std::vector<std::string> inputs;
  for (int i = 0; i < graph->node_size(); ++i) {
    if (graph->node(i).op_type() == "MultiClassNMS") {
      nms_index = -1;
      for (int j = 0; j < graph->node(i).input_size(); ++j) {
        inputs.push_back(graph->node(i).input(j));
      }
      break;
    }
  }
  graph->clear_output();
  for (size_t i = 0; i < inputs.size(); ++i) {
    auto output = graph->add_output();
    output->set_name(inputs[i]);
    auto type_proto = output->mutable_type();
    auto tensor_type_proto = type_proto->mutable_tensor_type();
    tensor_type_proto->set_elem_type(ONNX_NAMESPACE::TensorProto::FLOAT);
    auto shape = tensor_type_proto->mutable_shape();
    shape->add_dim()->set_dim_value(-1);
    shape->add_dim()->set_dim_value(-1);
    shape->add_dim()->set_dim_value(-1);
  }
  auto optimized_model = ONNX_NAMESPACE::optimization::OptimizeOnnxModel(model);
  *out_model_size = optimized_model.ByteSizeLong();
  *out_model = new char[*out_model_size];
  optimized_model.SerializeToArray(*out_model, *out_model_size);
  return true;
}

}  // namespace paddle2onnx
