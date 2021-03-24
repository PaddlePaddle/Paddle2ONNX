// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "include/deploy/engine/openvino_engine.h"

namespace Deploy {


void OpenVinoEngine::Init(const std::string &model_filename,
                          const OpenVinoEngineConfig &engine_config) {
  InferenceEngine::Core ie;
  network_ = ie.ReadNetwork(model_filename,
          model_filename.substr(0, model_filename.size() - 4) + ".bin");
  network_.setBatchSize(engine_config.batch_size);
  if (engine_config.device == "MYRIAD") {
    std::map<std::string, std::string> networkConfig;
    networkConfig["VPU_HW_STAGES_OPTIMIZATION"] = "NO";
    executable_network_ = ie.LoadNetwork(
            network_, engine_config.device, networkConfig);
  } else {
    executable_network_ = ie.LoadNetwork(network_, engine_config.device);
  }
}

void OpenVinoEngine::Infer(const std::vector<DataBlob> &inputs,
                          std::vector<DataBlob> *outputs) {
  InferRequest infer_request = executable_network_.CreateInferRequest();
  for (int i = 0; i < inputs.size(); i++) {
    std::vector<int> input_shape = inputs[i].shape;
    InferenceEngine::TensorDesc input_tensor;
    if (input_shape.size() == 4) {
      input_tensor.setLayout(InferenceEngine::Layout::NCHW);
    }
    if (inputs[i].dtype == 0) {
      input_tensor.setPrecision(InferenceEngine::Precision::FP32);
      InferenceEngine::Blob::Ptr input_blob =
            InferenceEngine::make_shared_blob<float>(input_tensor,
            reinterpret_cast<float*>(inputs[i].data.data());
      infer_request.SetBlob(inputs[i].name, input_blob);
    } else if (output.dtype == 1) {
      input_tensor.setPrecision(InferenceEngine::Precision::U64);
      InferenceEngine::Blob::Ptr input_blob =
            InferenceEngine::make_shared_blob<int64_t>(input_tensor,
            reinterpret_cast<int64_t*>(inputs[i].data.data());
      infer_request.SetBlob(inputs[i].name, input_blob);
    } else if (output.dtype == 2) {
      input_tensor.setPrecision(InferenceEngine::Precision::I32);
      InferenceEngine::Blob::Ptr input_blob =
            InferenceEngine::make_shared_blob<int>(input_tensor,
            reinterpret_cast<int*>(inputs[i].data.data());
      infer_request.SetBlob(inputs[i].name, input_blob);
    } else if (output.dtype == 3) {
      input_tensor.setPrecision(InferenceEngine::Precision::U8);
      InferenceEngine::Blob::Ptr input_blob =
            InferenceEngine::make_shared_blob<uint8_t>(input_tensor,
            reinterpret_cast<uint8_t*>(inputs[i].data.data());
      infer_request.SetBlob(inputs[i].name, input_blob);
    }
  }

  // do inference
  infer_request.Infer();

  InferenceEngine::OutputsDataMap out_maps = network_.getOutputsInfo();
  for (const auto & output_map : out_maps) {
    DataBlob output;
    std::string name = output_map.first;
    output.name = name;
    InferenceEngine::Blob::Ptr output = infer_request.GetBlob(outputName);
    InferenceEngine::MemoryBlob::CPtr moutput =
      InferenceEngine::as<InferenceEngine::MemoryBlob>(output);
    InferenceEngine::TensorDesc blob_output = moutput->getTensorDesc();
    std::vector<int> output_shape = blob_output.getDims();
    int size = 1;
    for (auto& i : output_shape) {
      size *= i;
    }
    output.shape.assign(output_shape.begin(), output_shape.end());
    GetDtype(blob_output, &output)
    auto moutputHolder = moutput->rmap();
    if (output.dtype == 0) {
      float* data = moutputHolder.as<float *>();
      memcpy(output.data.data(), data, size * sizeof(float));
    } else if (output.dtype == 1) {
      int64_t* data = moutputHolder.as<int64_t *>();
      memcpy(output.data.data(), data, size * sizeof(int64_t));
    } else if (output.dtype == 2) {
      int* data = moutputHolder.as<int *>();
      memcpy(output.data.data(), data, size * sizeof(int));
    } else if (output.dtype == 3) {
      uint8_t* data = moutputHolder.as<uint8_t *>();
      memcpy(output.data.data(), data, size * sizeof(uint8_t));
    }
    outputs->push_back(std::move(output));
  }
}

bool OpenVinoEngine::GetDtype(const InferenceEngine::TensorDesc &output_blob,
                          DataBlob *output) {
  InferenceEngine::Precision output_precision output_blob.getPrecision();
  if (output_precision == 10) {
    output->dtype = 0;
  } else if (output_precision == 73) {
    output->dtype = 1;
  } else if (output_precision == 70) {
    output->dtype = 2;
  } else if (output_precision == 40) {
    output->dtype = 3;
  } else {
    std::cout << "can't paser the precision type" << std::endl;
    return false;
  }
  return true;
}

}  //  namespace Deploy
