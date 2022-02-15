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
#include <onnx/checker.h>
#include <onnx/onnx_pb.h>
#include <algorithm>
#include <set>

#include "paddle2onnx/mapper/nn.h"
#include "paddle2onnx/parser/parser.h"

namespace paddle2onnx {

struct ModelExporter {
 private:
  std::vector<std::shared_ptr<ONNX_NAMESPACE::NodeProto>> parameters;
  std::vector<std::shared_ptr<ONNX_NAMESPACE::ValueInfoProto>> inputs;
  std::vector<std::shared_ptr<ONNX_NAMESPACE::ValueInfoProto>> outputs;
  OnnxHelper helper;

  void ExportParameters(const std::map<std::string, Weight>& params,
                        bool use_initializer = false);
  void ExportInputOutputs(const std::vector<TensorInfo>& input_infos,
                          const std::vector<TensorInfo>& output_infos);
  void ExportOp(const PaddleParser& parser, int32_t opset_version,
                int64_t block_id, int64_t op_id);
  // Get a proper opset version in range of [7, 15]
  // Also will check the model is convertable, this will include 2 parts
  //    1. is the op convert function implemented
  //    2. is the op convertable(some cases may not be able to convert)
  // If the model is not convertable, return -1
  int32_t GetMinOpset(const PaddleParser& parser, bool verbose = false);

  bool CheckIfOpSupported(const PaddleParser& parser,
                          std::set<std::string>* unsupported_ops,
                          bool enable_experimental_op);

 public:
  std::string Run(const PaddleParser& parser, int opset_version = 9,
                  bool auto_upgrade_opset = true, bool verbose = false,
                  bool enable_onnx_checker = true,
                  bool enable_experimental_op = false);
};

}  // namespace paddle2onnx
