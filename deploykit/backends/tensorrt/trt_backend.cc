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

#include "deploykit/backends/tensorrt/trt_backend.h"

namespace deploykit {

size_t TrtDataTypeSize(const nvinfer1::DataType& dtype) {
  if (dtype == nvinfer1::DataType::kFLOAT) {
    return sizeof(float);
  } else if (dtype == nvinfer1::DataType::kHALF) {
    return sizeof(float) / 2;
  } else if (dtype == nvinfer1::DataType::kINT8) {
    return sizeof(int8_t);
  } else if (dtype == nvinfer1::DataType::kINT32) {
    return sizeof(int32_t);
  }
  // kBOOL
  return sizeof(bool);
}

size_t GetPaddleDataType(const nvinfer1::DataType& dtype) {
  if (dtype == nvinfer1::DataType::kFLOAT) {
    return PaddleDataType::FP32;
  } else if (dtype == nvinfer1::DataType::kHALF) {
    return PaddleDataType::FP16;
  } else if (dtype == nvinfer1::DataType::kINT8) {
    return PaddleDataType::INT8;
  } else if (dtype == nvinfer1::DataType::kINT32) {
    return PaddleDataType::INT32;
  }
  // kBOOL
  return PaddleDataType::BOOL;
}

std::vector<int> toVec(const nvinfer1::Dims& dim) {
  std::vector<int> out(dim.d, dim.d + dim.nbDims);
  return out;
}

bool TrtBackend::InitFromTrt(const std::string& trt_engine_file) {
  if (initialized_) {
    KitLogger() << "TrtBackend is already initlized, cannot initialize again."
                << std::endl;
    return false;
  }
  std::ifstream fin(trt_engine_file, std::ios::binary | std::ios::in);
  if (!fin) {
    KitLogger() << "[ERROR] Failed to open TensorRT Engine file "
                << trt_engine_file << std::endl;
    return false;
  }
  fin.seekg(0, std::ios::end);
  std::string engine_buffer;
  engine_buffer.resize(fin.tellg());
  fin.seekg(0, std::ios::beg);
  fin.read(&(engine_buffer.at(0)), engine_buffer.size());
  fin.close();
  SampleUniquePtr<IRuntime> runtime{
      createInferRuntime(sample::gLogger.getTRTLogger())};
  if (!runtime) {
    KitLogger() << "[ERROR] Failed to call createInferRuntime()." << std::endl;
    return false;
  }
  engine_ = std::shared_ptr<nvinfer1::ICudaEngine>(
      runtime->deserializeCudaEngine(engine_buffer.data(),
                                     engine_buffer.size()),
      samplesCommon::InferDeleter());
  if (!engine_) {
    KitLogger() << "[ERROR] Failed to call deserializeCudaEngine()."
                << std::endl;
    return false;
  }

  context_ = std::shared_ptr<nvinfer1::IExecutionContext>(
      engine_->createExecutionContext());
  Assert(cudaStreamCreate(&stream_) == 0,
         "[ERROR] Error occurs while calling cudaStreamCreate().");
  GetInputOutputInfo();
  initialized_ = true;
  return true;
}

bool TrtBackend::InitFromPaddle(const std::string& model_file,
                                const std::string& params_file,
                                const TrtBackendOption& option, bool verbose) {
  if (initialized_) {
    KitLogger() << "TrtBackend is already initlized, cannot initialize again."
                << std::endl;
    return false;
  }

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
  return false;
#endif
}

bool TrtBackend::InitFromOnnx(const std::string& model_file,
                              const TrtBackendOption& option,
                              bool from_memory_buffer) {
  if (initialized_) {
    KitLogger() << "TrtBackend is already initlized, cannot initialize again."
                << std::endl;
    return false;
  }
  cudaSetDevice(option.gpu_id);

  if (option.serialize_file != "") {
    std::ifstream fin(option.serialize_file, std::ios::binary | std::ios::in);
    if (fin) {
      KitLogger() << "Detect serialized TensorRT Engine file in "
                  << option.serialize_file << ", will load it directly."
                  << std::endl;
      fin.close();
      return InitFromTrt(option.serialize_file);
    }
  }

  std::string onnx_content = "";
  if (!from_memory_buffer) {
    std::ifstream fin(model_file.c_str(), std::ios::binary | std::ios::in);
    if (!fin) {
      KitLogger() << "[ERROR] Failed to open ONNX model file: " << model_file
                  << std::endl;
      return false;
    }
    fin.seekg(0, std::ios::end);
    onnx_content.resize(fin.tellg());
    fin.seekg(0, std::ios::beg);
    fin.read(&(onnx_content.at(0)), onnx_content.size());
    fin.close();
  } else {
    onnx_content = model_file;
  }

  if (!CreateTrtEngine(onnx_content, option)) {
    return false;
  }

  context_ = std::shared_ptr<nvinfer1::IExecutionContext>(
      engine_->createExecutionContext());
  Assert(cudaStreamCreate(&stream_) == 0,
         "[ERROR] Error occurs while calling cudaStreamCreate().");
  GetInputOutputInfo();
  initialized_ = true;
  return true;
}

bool TrtBackend::Infer(std::vector<DataBlob>& inputs,
                       std::vector<DataBlob>* outputs) {
  AllocateBufferInDynamicShape(inputs, outputs);
  std::vector<void*> input_binds(inputs.size());
  for (size_t i = 0; i < inputs.size(); ++i) {
    if (inputs[0].dtype == PaddleDataType::INT64) {
      int64_t* data = static_cast<int64_t*>(inputs[i].GetData());
      std::vector<int32_t> casted_data(data, data + inputs[i].Numel());
      Assert(cudaMemcpyAsync(inputs_buffer_[inputs[i].name].data(), static_cast<void*>(casted_data.data()), inputs[i].Nbytes() / 2, cudaMemcpyHostToDevice, stream_) == 0, "[ERROR] Error occurs while copy memory from CPU to GPU.");
    } else {
      Assert(cudaMemcpyAsync(inputs_buffer_[inputs[i].name].data(),
                             inputs[i].GetData(), inputs[i].Nbytes(),
                             cudaMemcpyHostToDevice, stream_) == 0,
             "[ERROR] Error occurs while copy memory from CPU to GPU.");
    }
    //    Assert(cudaMemcpy(inputs_buffer_[inputs[i].name].data(),
    //                           inputs[i].GetData(), inputs[i].Nbytes(),
    //                           cudaMemcpyHostToDevice) == 0,
    //           "[ERROR] Error occurs while copy memory from CPU to GPU.");
  }
  if (!context_->enqueueV2(bindings_.data(), stream_, nullptr)) {
    KitLogger() << "Failed to Infer with TensorRT." << std::endl;
    return false;
  }
  for (size_t i = 0; i < outputs->size(); ++i) {
    Assert(cudaMemcpyAsync((*outputs)[i].data.data(),
                           outputs_buffer_[(*outputs)[i].name].data(),
                           (*outputs)[i].Nbytes(), cudaMemcpyDeviceToHost,
                           stream_) == 0,
           "[ERROR] Error occurs while copy memory from GPU to CPU.");
    //    Assert(cudaMemcpy((*outputs)[i].data.data(),
    //                           outputs_buffer_[(*outputs)[i].name].data(),
    //                           (*outputs)[i].Nbytes(),
    //                           cudaMemcpyDeviceToHost) == 0,
    //           "[ERROR] Error occurs while copy memory from GPU to CPU.");
  }
  //  Assert(cudaStreamSynchronize(stream_) == 0,
  //         "[ERROR] Error occurs while calling cudaStreamSynchronize().");
  return true;
}

void TrtBackend::GetInputOutputInfo() {
  inputs_desc_.clear();
  outputs_desc_.clear();
  auto num_binds = engine_->getNbBindings();
  for (auto i = 0; i < num_binds; ++i) {
    std::string name = std::string(engine_->getBindingName(i));
    auto shape = toVec(engine_->getBindingDimensions(i));
    auto dtype = engine_->getBindingDataType(i);
    if (engine_->bindingIsInput(i)) {
      inputs_desc_.emplace_back(TrtValueInfo{name, shape, dtype});
      inputs_buffer_[name] = DeviceBuffer(dtype);
    } else {
      outputs_desc_.emplace_back(TrtValueInfo{name, shape, dtype});
      outputs_buffer_[name] = DeviceBuffer(dtype);
    }
  }
  bindings_.resize(num_binds);
}

void TrtBackend::AllocateBufferInDynamicShape(
    const std::vector<DataBlob>& inputs, std::vector<DataBlob>* outputs) {
  for (const auto& item : inputs) {
    auto idx = engine_->getBindingIndex(item.name.c_str());
    std::vector<int> shape(item.shape.begin(), item.shape.end());
    auto dims = sample::toDims(shape);
    context_->setBindingDimensions(idx, dims);
    if (item.Nbytes() > inputs_buffer_[item.name].nbBytes()) {
      inputs_buffer_[item.name].resize(dims);
      bindings_[idx] = inputs_buffer_[item.name].data();
    }
  }
  if (outputs->size() != outputs_desc_.size()) {
    outputs->resize(outputs_desc_.size());
  }
  for (size_t i = 0; i < outputs_desc_.size(); ++i) {
    auto idx = engine_->getBindingIndex(outputs_desc_[i].name.c_str());
    auto output_dims = context_->getBindingDimensions(idx);
    (*outputs)[i].dtype = GetPaddleDataType(outputs_desc_[i].dtype);
    (*outputs)[i].shape.assign(output_dims.d,
                               output_dims.d + output_dims.nbDims);
    (*outputs)[i].name = outputs_desc_[i].name;
    (*outputs)[i].data.resize(volume(output_dims) *
                              TrtDataTypeSize(outputs_desc_[i].dtype));
    if ((*outputs)[i].Nbytes() >
        outputs_buffer_[outputs_desc_[i].name].nbBytes()) {
      outputs_buffer_[outputs_desc_[i].name].resize(output_dims);
      bindings_[idx] = outputs_buffer_[outputs_desc_[i].name].data();
    }
  }
}

bool TrtBackend::CreateTrtEngine(const std::string& onnx_model,
                                 const TrtBackendOption& option) {
  const auto explicitBatch =
      1U << static_cast<uint32_t>(
          nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);

  auto builder = SampleUniquePtr<nvinfer1::IBuilder>(
      nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger()));
  if (!builder) {
    KitLogger() << "[ERROR] Failed to call createInferBuilder()." << std::endl;
    return false;
  }
  auto network = SampleUniquePtr<nvinfer1::INetworkDefinition>(
      builder->createNetworkV2(explicitBatch));
  if (!network) {
    KitLogger() << "[ERROR] Failed to call createNetworkV2()." << std::endl;
    return false;
  }
  auto config =
      SampleUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
  if (!config) {
    KitLogger() << "[ERROR] Failed to call createBuilderConfig()." << std::endl;
    return false;
  }
  auto parser = SampleUniquePtr<nvonnxparser::IParser>(
      nvonnxparser::createParser(*network, sample::gLogger.getTRTLogger()));
  if (!parser) {
    KitLogger() << "[ERROR] Failed to call createParser()." << std::endl;
    return false;
  }
  if (!parser->parse(onnx_model.data(), onnx_model.size())) {
    KitLogger() << "[ERROR] Failed to parse ONNX model by TensorRT."
                << std::endl;
    return false;
  }

  KitLogger() << "Start to building TensorRT Engine..." << std::endl;
  bool fp16 = builder->platformHasFastFp16();
  builder->setMaxBatchSize(option.max_batch_size);

  config->setMaxWorkspaceSize(option.max_workspace_size);

  if (option.fixed_shape.size() > 0) {
    auto profile = builder->createOptimizationProfile();
    for (auto& item : option.fixed_shape) {
      Assert(profile->setDimensions(item.first.c_str(),
                                    nvinfer1::OptProfileSelector::kMIN,
                                    sample::toDims(item.second)),
             "[TrtBackend] Failed to set min_shape for input: " + item.first +
                 " in TrtBackend.");
      Assert(profile->setDimensions(item.first.c_str(),
                                    nvinfer1::OptProfileSelector::kOPT,
                                    sample::toDims(item.second)),
             "[TrtBackend] Failed to set min_shape for input: " + item.first +
                 " in TrtBackend.");
      Assert(profile->setDimensions(item.first.c_str(),
                                    nvinfer1::OptProfileSelector::kMAX,
                                    sample::toDims(item.second)),
             "[TrtBackend] Failed to set min_shape for input: " + item.first +
                 " in TrtBackend.");
    }
    config->addOptimizationProfile(profile);
  } else if (option.max_shape.size() > 0) {
    auto profile = builder->createOptimizationProfile();
    Assert(option.max_shape.size() == option.min_shape.size() &&
               option.min_shape.size() == option.opt_shape.size(),
           "[TrtBackend] Size of max_shape/opt_shape/min_shape in "
           "TrtBackendOption should keep same.");
    for (const auto& item : option.min_shape) {
      // set min shape
      Assert(profile->setDimensions(item.first.c_str(),
                                    nvinfer1::OptProfileSelector::kMIN,
                                    sample::toDims(item.second)),
             "[TrtBackend] Failed to set min_shape for input: " + item.first +
                 " in TrtBackend.");

      // set optimization shape
      auto iter = option.opt_shape.find(item.first);
      Assert(iter != option.opt_shape.end(),
             "[TrtBackend] Cannot find input name: " + item.first +
                 " in TrtBackendOption::opt_shape.");
      Assert(profile->setDimensions(item.first.c_str(),
                                    nvinfer1::OptProfileSelector::kOPT,
                                    sample::toDims(iter->second)),
             "[TrtBackend] Failed to set opt_shape for input: " + item.first +
                 " in TrtBackend.");
      // set max shape
      iter = option.max_shape.find(item.first);
      Assert(iter != option.max_shape.end(),
             "[TrtBackend] Cannot find input name: " + item.first +
                 " in TrtBackendOption::max_shape.");
      Assert(profile->setDimensions(item.first.c_str(),
                                    nvinfer1::OptProfileSelector::kMAX,
                                    sample::toDims(iter->second)),
             "[TrtBackend] Failed to set max_shape for input: " + item.first +
                 " in TrtBackend.");
    }
    config->addOptimizationProfile(profile);
  }

  SampleUniquePtr<IHostMemory> plan{
      builder->buildSerializedNetwork(*network, *config)};
  if (!plan) {
    KitLogger() << "[ERROR] Failed to call buildSerializedNetwork()."
                << std::endl;
    return false;
  }

  SampleUniquePtr<IRuntime> runtime{
      createInferRuntime(sample::gLogger.getTRTLogger())};
  if (!runtime) {
    KitLogger() << "[ERROR] Failed to call createInferRuntime()." << std::endl;
    return false;
  }

  engine_ = std::shared_ptr<nvinfer1::ICudaEngine>(
      runtime->deserializeCudaEngine(plan->data(), plan->size()),
      samplesCommon::InferDeleter());
  if (!engine_) {
    KitLogger() << "[ERROR] Failed to call deserializeCudaEngine()."
                << std::endl;
    return false;
  }

  KitLogger() << "TensorRT Engine is built succussfully." << std::endl;
  if (option.serialize_file != "") {
    KitLogger() << "Serialize TensorRTEngine to local file "
                << option.serialize_file << "." << std::endl;
    std::ofstream engine_file(option.serialize_file.c_str());
    if (!engine_file) {
      KitLogger() << "[ERROR] Failed to open " << option.serialize_file
                  << " to write." << std::endl;
      return false;
    }
    engine_file.write(static_cast<char*>(plan->data()), plan->size());
    engine_file.close();
    KitLogger() << "TensorRTEngine is serialized to local file "
                << option.serialize_file
                << ", we can load this model from the seralized engine "
                   "directly next time."
                << std::endl;
  }
  return true;
}
}  // namespace deploykit
