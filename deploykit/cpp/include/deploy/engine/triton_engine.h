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

#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <memory>

#include "./common.h"
#include "./http_client.h"
#include "rapidjson/document.h"
#include "rapidjson/rapidjson.h"

#include "include/deploy/common/blob.h"

namespace nic = nvidia::inferenceserver::client;

namespace Deploy {

struct TritonInferenceConfigs {
  explicit TritonInferenceConfigs(const std::string &model_name)
      : model_name_(model_name), model_version_(""), request_id_(""),
        sequence_id_(0), sequence_start_(false), sequence_end_(false),
        priority_(0), server_timeout_(0), client_timeout_(0) {}
  /// The name of the model to run inference.
  std::string model_name_;
  /// The version of the model to use while running inference. The default
  /// value is an empty string which means the server will select the
  /// version of the model based on its internal policy.
  std::string model_version_;
  /// An identifier for the request. If specified will be returned
  /// in the response. Default value is an empty string which means no
  /// request_id will be used.
  std::string request_id_;
  /// The unique identifier for the sequence being represented by the
  /// object. Default value is 0 which means that the request does not
  /// belong to a sequence.
  uint64_t sequence_id_;
  /// Indicates whether the request being added marks the start of the
  /// sequence. Default value is False. This argument is ignored if
  /// 'sequence_id' is 0.
  bool sequence_start_;
  /// Indicates whether the request being added marks the end of the
  /// sequence. Default value is False. This argument is ignored if
  /// 'sequence_id' is 0.
  bool sequence_end_;
  /// Indicates the priority of the request. Priority value zero
  /// indicates that the default priority level should be used
  /// (i.e. same behavior as not specifying the priority parameter).
  /// Lower value priorities indicate higher priority levels. Thus
  /// the highest priority level is indicated by setting the parameter
  /// to 1, the next highest is 2, etc. If not provided, the server
  /// will handle the request using default setting for the model.
  uint64_t priority_;
  /// The timeout value for the request, in microseconds. If the request
  /// cannot be completed within the time by the server can take a
  /// model-specific action such as terminating the request. If not
  /// provided, the server will handle the request using default setting
  /// for the model.
  uint64_t server_timeout_;
  // The maximum end-to-end time, in microseconds, the request is allowed
  // to take. Note the HTTP library only offer the precision upto
  // milliseconds. The client will abort request when the specified time
  // elapses. The request will return error with message "Deadline Exceeded".
  // The default value is 0 which means client will wait for the
  // response from the server. This option is not supported for streaming
  // requests. Instead see 'stream_timeout' argument in
  // InferenceServerGrpcClient::StartStream().
  uint64_t client_timeout_;
};

class TritonInferenceEngine {
 public:
  void Init(const std::string &url, bool verbose = false);

  void Infer(const TritonInferenceConfigs &configs,
             const std::vector<DataBlob> &input_blobs,
             std::vector<DataBlob> *output_blobs,
             const nic::Headers &headers = nic::Headers(),
             const nic::Parameters &query_params = nic::Parameters());

  std::unique_ptr<nic::InferenceServerHttpClient> client_;

 private:
  void ParseConfigs(const TritonInferenceConfigs &configs,
                    nic::InferOptions *options);

  void CreateInput(const std::vector<DataBlob> &input_blobs,
                   std::vector<nic::InferInput *> *inputs);

  void CreateOutput(const rapidjson::Document &model_metadata,
                    std::vector<const nic::InferRequestedOutput *> *outputs);

  nic::Error GetModelMetaData(const std::string &model_name,
                              const std::string &model_version,
                              const nic::Headers &http_headers,
                              rapidjson::Document *model_metadata);
};

}  // namespace Deploy
