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

#pragma once

#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "deploykit/backends/data_blob.h"
#include "deploykit/utils/utils.h"

#ifdef ENABLE_PADDLE_FRONTEND
#include "paddle2onnx/converter.h"
#endif

#include "deploykit/backends/tensorrt/common/argsParser.h"
#include "deploykit/backends/tensorrt/common/buffers.h"
#include "deploykit/backends/tensorrt/common/common.h"
#include "deploykit/backends/tensorrt/common/logger.h"
#include "deploykit/backends/tensorrt/common/parserOnnxConfig.h"
#include "deploykit/backends/tensorrt/common/sampleUtils.h"

#include <cuda_runtime_api.h>
#include "NvInfer.h"

namespace deploykit {

using namespace samplesCommon;

struct TrtValueInfo {
  std::string name;
  std::vector<int> shape;
  nvinfer1::DataType dtype;
};

struct TrtBackendOption {
  int gpu_id = 0;
  int enable_fp16 = 0;  // 0 or 1
  int enable_int8 = 0;  // 0 or 1
  size_t max_batch_size = 32;
  size_t max_workspace_size = 1 << 30;
  std::map<std::string, std::vector<int32_t>> fixed_shape;
  std::map<std::string, std::vector<int32_t>> max_shape;
  std::map<std::string, std::vector<int32_t>> min_shape;
  std::map<std::string, std::vector<int32_t>> opt_shape;
  std::string serialize_file = "";
};

std::vector<int> toVec(const nvinfer1::Dims& dim);
size_t TrtDataTypeSize(const nvinfer1::DataType& dtype);
size_t GetPaddleDataType(const nvinfer1::DataType& dtype);

class TrtBackend {
 public:
  TrtBackend() : engine_(nullptr), context_(nullptr) {}
  void BuildOption(const TrtBackendOption& option);

  bool InitFromPaddle(const std::string& model_file,
                      const std::string& params_file,
                      const TrtBackendOption& option = TrtBackendOption(),
                      bool verbose = false);
  bool InitFromOnnx(const std::string& model_file,
                    const TrtBackendOption& option = TrtBackendOption(),
                    bool from_memory_buffer = false);
  bool InitFromTrt(const std::string& trt_engine_file);

  bool Infer(std::vector<DataBlob>& inputs, std::vector<DataBlob>* outputs);

  int NumInputs() const { return inputs_desc_.size(); }
  int NumOutputs() const { return outputs_desc_.size(); }
  bool Initliazed() const { return initialized_; }

  //  ~TrtBackend() {
  //    Assert(cudaStreamDestroy(stream_) == 0, "Failed to call
  //    cudaStreamDestroy().");
  //  }

 private:
  bool initialized_ = false;
  std::shared_ptr<nvinfer1::ICudaEngine> engine_;
  std::shared_ptr<nvinfer1::IExecutionContext> context_;
  cudaStream_t stream_{};
  std::vector<void*> bindings_;
  std::vector<TrtValueInfo> inputs_desc_;
  std::vector<TrtValueInfo> outputs_desc_;
  std::map<std::string, DeviceBuffer> inputs_buffer_;
  std::map<std::string, DeviceBuffer> outputs_buffer_;

  void GetInputOutputInfo();
  void AllocateBufferInDynamicShape(const std::vector<DataBlob>& inputs,
                                    std::vector<DataBlob>* outputs);
  bool CreateTrtEngine(const std::string& onnx_model,
                       const TrtBackendOption& option);
};
}  // namespace deploykit
