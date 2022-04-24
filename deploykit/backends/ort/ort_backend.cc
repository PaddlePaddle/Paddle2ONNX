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

#include "deploykit/backends/ort/ort_backend.h"
#include <memory>
#ifdef ENABLE_PADDLE_FRONTEND
#include "paddle2onnx/converter.h"
#endif

namespace deploykit {

ONNXTensorElementDataType GetOrtDtype(int paddle_dtype) {
  if (paddle_dtype == PaddleDataType::FP32) {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  } else if (paddle_dtype == PaddleDataType::FP64) {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE;
  } else if (paddle_dtype == PaddleDataType::INT32) {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
  } else if (paddle_dtype == PaddleDataType::INT64) {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
  }
  KitLogger() << "Unrecognized paddle data type:" << paddle_dtype
              << " while calling GetOrtDtype()." << std::endl;
  return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
}

void OrtBackend::BuildOption(const OrtBackendOption& option) {
  if (option.graph_optimization_level >= 0) {
    session_options_.SetGraphOptimizationLevel(
        GraphOptimizationLevel(option.graph_optimization_level));
  }
  if (option.intra_op_num_threads >= 0) {
    session_options_.SetIntraOpNumThreads(option.intra_op_num_threads);
  }
  if (option.inter_op_num_threads >= 0) {
    session_options_.SetInterOpNumThreads(option.inter_op_num_threads);
  }
  if (option.execution_mode >= 0) {
    session_options_.SetExecutionMode(ExecutionMode(option.execution_mode));
  }
}

bool OrtBackend::InitFromPaddle(const std::string& model_file,
                                const std::string& params_file,
                                const OrtBackendOption& option, bool verbose) {
#ifdef ENABLE_PADDLE_FRONTEND
  std::string onnx_model_proto;
  if (!paddle2onnx::Export(model_file, params_file, &onnx_model_proto, false,
                           11, true, verbose, true, true, true)) {
    KitLogger() << "Error occured while export PaddlePaddle to ONNX format."
                << std::endl;
    return false;
  }
  return InitFromOnnx(onnx_model_proto, option, true);
#else
  KitLogger() << "Didn't compile with PaddlePaddle frontend, you can try to "
                 "call `InitFromOnnx` instead."
              << std::endl;
#endif
  return false;
}

bool OrtBackend::InitFromOnnx(const std::string& model_file,
                              const OrtBackendOption& option,
                              bool from_memory_buffer) {
  if (initialized_) {
    KitLogger() << "OrtBackend is already initlized, cannot initialize again."
                << std::endl;
    return false;
  }
  BuildOption(option);
  if (from_memory_buffer) {
    session_ = {env_, model_file.data(), model_file.size(), session_options_};
  } else {
#ifdef _WIN32
    session_ = {env_,
                std::wstring(model_file.begin(), model_file.end()).c_str(),
                session_options_};
#else
    session_ = {env_, model_file.c_str(), session_options_};
#endif
  }
  binding_ = std::make_shared<Ort::IoBinding>(session_);

  Ort::MemoryInfo memory_info("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault);
  Ort::Allocator allocator(session_, memory_info);
  size_t n_inputs = session_.GetInputCount();
  for (size_t i = 0; i < n_inputs; ++i) {
    auto input_name = session_.GetInputName(i, allocator);
    auto type_info = session_.GetInputTypeInfo(i);
    std::vector<int64_t> shape =
        type_info.GetTensorTypeAndShapeInfo().GetShape();
    ONNXTensorElementDataType data_type =
        type_info.GetTensorTypeAndShapeInfo().GetElementType();
    inputs_desc_.emplace_back(OrtValueInfo{input_name, shape, data_type});
    allocator.Free(input_name);
  }

  size_t n_outputs = session_.GetOutputCount();
  for (size_t i = 0; i < n_outputs; ++i) {
    auto output_name = session_.GetOutputName(i, allocator);
    auto type_info = session_.GetOutputTypeInfo(i);
    std::vector<int64_t> shape =
        type_info.GetTensorTypeAndShapeInfo().GetShape();
    ONNXTensorElementDataType data_type =
        type_info.GetTensorTypeAndShapeInfo().GetElementType();
    outputs_desc_.emplace_back(OrtValueInfo{output_name, shape, data_type});

    Ort::MemoryInfo out_memory_info("Cpu", OrtDeviceAllocator, 0,
                                    OrtMemTypeDefault);
    binding_->BindOutput(output_name, out_memory_info);

    allocator.Free(output_name);
  }
  initialized_ = true;
  return true;
}

void OrtBackend::CopyToCpu(const Ort::Value& value, DataBlob* blob) {
  const auto info = value.GetTensorTypeAndShapeInfo();
  const auto data_type = info.GetElementType();
  size_t numel = info.GetElementCount();
  blob->shape = info.GetShape();

  if (data_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    blob->data.resize(numel * sizeof(float));
    memcpy(static_cast<void*>(blob->GetData()), value.GetTensorData<void*>(),
           numel * sizeof(float));
    blob->dtype = PaddleDataType::FP32;
  } else if (data_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
    blob->data.resize(numel * sizeof(int32_t));
    memcpy(static_cast<void*>(blob->GetData()), value.GetTensorData<void*>(),
           numel * sizeof(int32_t));
    blob->dtype = PaddleDataType::INT32;
  } else if (data_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
    blob->data.resize(numel * sizeof(int64_t));
    memcpy(static_cast<void*>(blob->GetData()), value.GetTensorData<void*>(),
           numel * sizeof(int64_t));
    blob->dtype = PaddleDataType::INT64;
  } else if (data_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE) {
    blob->data.resize(numel * sizeof(double));
    memcpy(static_cast<void*>(blob->GetData()), value.GetTensorData<void*>(),
           numel * sizeof(double));
    blob->dtype = PaddleDataType::FP64;
  } else {
    Assert(false, "Unrecognized data type of " + std::to_string(data_type) +
                      " while calling OrtBackend::CopyToCpu().");
  }
}

bool OrtBackend::Infer(const std::vector<DataBlob>& inputs,
                       std::vector<DataBlob>* outputs) {
  if (inputs.size() != inputs_desc_.size()) {
    KitLogger() << "[OrtBackend] Size of the inputs(" << inputs.size()
                << ") should keep same with the inputs of this model("
                << inputs_desc_.size() << ")." << std::endl;
    return false;
  }

  std::cout << "11111111" << std::endl;
  // Copy from DataBlob to Ort Inputs
  Ort::MemoryInfo memory_info("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault);
  for (size_t i = 0; i < inputs.size(); ++i) {
    size_t data_byte_count =
        inputs[i].Numel() * PaddleDataTypeSize(inputs[i].dtype);
    auto ort_dtype = GetOrtDtype(inputs[i].dtype);
    auto ort_value = Ort::Value::CreateTensor(
        memory_info, inputs[i].GetData(), data_byte_count,
        inputs[i].shape.data(), inputs[i].shape.size(), ort_dtype);
    binding_->BindInput(inputs[i].name.c_str(), ort_value);
  }
  std::cout << "11111112" << std::endl;

  // Inference with inputs
  try {
    session_.Run({}, *(binding_.get()));
  } catch (const std::exception& e) {
    KitLogger() << "Failed to Infer: " << e.what() << std::endl;
    return false;
  }

  std::cout << "11111113" << std::endl;

  // Copy result after inference
  std::vector<Ort::Value> ort_outputs = binding_->GetOutputValues();
  outputs->resize(ort_outputs.size());
  for (size_t i = 0; i < ort_outputs.size(); ++i) {
    std::cout << "1111114 " << i << std::endl;
    (*outputs)[i].name = outputs_desc_[i].name;
    CopyToCpu(ort_outputs[i], &((*outputs)[i]));
  }
  return true;
}
}  // namespace deploykit
