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
#include <onnx/onnx_pb.h>
#include <fstream>

#include "paddle2onnx/mapper/mapper.h"
#include "paddle2onnx/parser/parser.h"
namespace paddle2onnx {

struct QuantizeModelProcessor {
 public:
  std::vector<QuantizeInfo> quantize_info;
  // Convert to different model formats based on backend, backend can be
  // TensorRT, ONNXRuntime and Others
  void ProcessQuantizeModel(
      std::vector<std::shared_ptr<ONNX_NAMESPACE::NodeProto>>* parameters,
      std::vector<std::shared_ptr<ONNX_NAMESPACE::ValueInfoProto>>* inputs,
      std::vector<std::shared_ptr<ONNX_NAMESPACE::ValueInfoProto>>* outputs,
      std::vector<std::shared_ptr<ONNX_NAMESPACE::NodeProto>>* nodes,
      OnnxHelper& helper, const std::string& deploy_backend);

  // Remove all Quantize and Dequantize ops
  void RemoveAllQuantizeOps(
      std::vector<std::shared_ptr<ONNX_NAMESPACE::NodeProto>>* parameters,
      std::vector<std::shared_ptr<ONNX_NAMESPACE::ValueInfoProto>>* inputs,
      std::vector<std::shared_ptr<ONNX_NAMESPACE::ValueInfoProto>>* outputs,
      std::vector<std::shared_ptr<ONNX_NAMESPACE::NodeProto>>* nodes,
      OnnxHelper& helper);

  // Generate name2node_dict to save input name and its related nodes
  void InputNameToNodes(
      const std::vector<std::shared_ptr<ONNX_NAMESPACE::NodeProto>>& nodes,
      std::map<std::string,
               std::vector<std::shared_ptr<ONNX_NAMESPACE::NodeProto>>>*
          name2node_dict);

  void RemoveNodeByName(
      const std::map<std::string,
                     std::vector<std::shared_ptr<ONNX_NAMESPACE::NodeProto>>>&
          name2node_dict,
      std::vector<std::shared_ptr<ONNX_NAMESPACE::NodeProto>>* nodes,
      const std::string& name);

  void ReplaceInputOfAllNodes(
      const std::map<std::string,
                     std::vector<std::shared_ptr<ONNX_NAMESPACE::NodeProto>>>&
          name2node_dict,
      const std::string& old_name, const std::string& new_name);
};
}  // namespace paddle2onnx
