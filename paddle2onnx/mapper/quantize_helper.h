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
#include <cmath>
#include <fstream>

#include "paddle2onnx/mapper/mapper.h"
#include "paddle2onnx/parser/parser.h"
namespace paddle2onnx {

struct QuantizeModelProcessor {
 public:
  std::vector<QuantizeInfo> quantize_info;
  const PaddleParser* parser_;
  OnnxHelper* helper_;

  std::vector<std::shared_ptr<ONNX_NAMESPACE::NodeProto>>* parameters_;
  std::vector<std::shared_ptr<ONNX_NAMESPACE::ValueInfoProto>>* inputs_;
  std::vector<std::shared_ptr<ONNX_NAMESPACE::ValueInfoProto>>* outputs_;
  std::vector<std::shared_ptr<ONNX_NAMESPACE::NodeProto>>* nodes_;

  std::map<std::string, std::vector<std::shared_ptr<ONNX_NAMESPACE::NodeProto>>>
      name2node_dict_;
  std::vector<std::string> tensors_to_be_quantize;  // records those tensors
                                                    // that need to add quantize
                                                    // and dequantize op
  std::vector<std::string> only_dequantize_tensors;  // records those tensors
                                                     // that only need to add
                                                     // the dequantize op
  // Convert to different model formats based on backend, backend can be
  // TensorRT, ONNXRuntime and Others
  void ProcessQuantizeModel(
      std::vector<std::shared_ptr<ONNX_NAMESPACE::NodeProto>>* parameters,
      std::vector<std::shared_ptr<ONNX_NAMESPACE::ValueInfoProto>>* inputs,
      std::vector<std::shared_ptr<ONNX_NAMESPACE::ValueInfoProto>>* outputs,
      std::vector<std::shared_ptr<ONNX_NAMESPACE::NodeProto>>* nodes,
      OnnxHelper* helper, const std::string& deploy_backend,
      const PaddleParser& parser);

  // Remove all Quantize and Dequantize ops
  void RemoveAllQuantizeOps();

  // If all tensors in tensor_names have quantize info, return True, otherwise
  // return false
  bool CanBeQuantize(const std::vector<std::string>& tensor_names);
  // only_dequantize records those tensors that only need to add the dequantize
  // op
  void AppendQuantizeTensor(const std::string& tensor,
                            const bool& only_dequantize = false);

  // According to:
  // https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/core/optimizer/qdq_transformer/selectors_actions/qdq_selector_action_transformer.cc
  void AddQDQ();

  void QuantizeInfoBroadcast();

  // merge conv + add
  void MergeConvAdd();

  // merge conv + BN
  void MergeConvBN();

  bool IsGraphOutput(const std::string& name);

  // Because processing the quantize model will add new nodes, which will
  // destroy the topo sorting of nodes, this function will sort the nodes again
  void SortNodes();

  bool GetTensorShape(const std::string& name, std::vector<int64_t>* shape);

  // return the value of tensor by name
  template <typename T>
  bool GetTensorByName(const std::string& name, std::vector<T>* value);

  // Perform tensor wise quantization, returning scale and zero
  void GetTensorWiseQuantizeInfo(const std::vector<float>& tensor,
                                 std::vector<float>* scale,
                                 std::vector<int64_t>* zero);

  // Perform channel wise quantization, returning scale and zero
  void GetChannelWiseQuantizeInfo(const std::vector<float>& tensor,
                                  const std::vector<int64_t>& shape,
                                  const int64_t& quant_axis,
                                  std::vector<float>* scale,
                                  std::vector<int64_t>* zero);

  // Generate name2node_dict to save input name and its related nodes
  void UpdateInputNameToNodes();

  void RemoveNodeByName(const std::string& name, const bool& update_io = true);

  void ReplaceInputOfAllNodes(const std::string& old_name,
                              const std::string& new_name);
};
}  // namespace paddle2onnx
