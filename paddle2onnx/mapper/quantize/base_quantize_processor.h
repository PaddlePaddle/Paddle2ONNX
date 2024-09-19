// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
#include <iomanip>

#include "paddle2onnx/mapper/mapper.h"
#include "paddle2onnx/parser/parser.h"

namespace paddle2onnx {
class BaseQuantizeProcessor {
 public:
  BaseQuantizeProcessor() = default;
  virtual ~BaseQuantizeProcessor() = default;

  std::vector<std::string> tensors_to_be_quantize;
  std::vector<std::string> only_dequantize_tensors;

  // Convert to different model formats based on backend, backend can be
  // TensorRT, ONNXRuntime and Others
  virtual void ProcessQuantizeModel(
      std::vector<std::shared_ptr<ONNX_NAMESPACE::NodeProto>> *parameters,
      std::vector<std::shared_ptr<ONNX_NAMESPACE::ValueInfoProto>> *inputs,
      std::vector<std::shared_ptr<ONNX_NAMESPACE::ValueInfoProto>> *outputs,
      std::vector<std::shared_ptr<ONNX_NAMESPACE::NodeProto>> *nodes,
      OnnxHelper *helper, const PaddleParser &parser,
      std::string *calibration_cache = nullptr);

  // If all tensors in tensor_names have quantize info and all the next nodes
  // can be quantized, return True, otherwise
  // return false
  bool CanBeQuantize(const std::vector<std::string> &tensor_names,
                     const std::vector<int64_t> &output_index = {-1});
  // only_dequantize records those tensors that only need to add the dequantize
  // op
  void AppendQuantizeTensor(const std::string &tensor,
                            const bool &only_dequantize = false);

  // Determine if the tensor is directly linked to the output by identity
  bool ConnectToOutput(const std::string &output_name);

  void QuantizeInfoBroadcast();
  void RemoveAllQuantizeOps();
  void MergeConvAdd();
  void MergeConvBN();

  void RemoveIdentityOp();

  // Add quantize related op in model according to tensor names
  void AddQDQInModel(const std::vector<std::string> &tensors_to_be_quantize);

  // Determine whether a tensor is an output
  bool IsGraphOutput(const std::string &name);

  // Because processing the quantize model will add new nodes, which will
  // destroy the topo sorting of nodes, this function will sort the nodes again
  void SortNodes();

  bool GetTensorShape(const std::string &name, std::vector<int64_t> *shape);

  // return the value of tensor by name
  template <typename T>
  bool GetTensorByName(const std::string &name, std::vector<T> *value);

  // Perform tensor wise quantization, returning scale and zero
  void GetTensorWiseQuantizeInfo(const std::vector<float> &tensor,
                                 std::vector<float> *scale,
                                 std::vector<int64_t> *zero);

  // Perform channel wise quantization, returning scale and zero
  void GetChannelWiseQuantizeInfo(const std::vector<float> &tensor,
                                  const std::vector<int64_t> &shape,
                                  const int64_t &quant_axis,
                                  std::vector<float> *scale,
                                  std::vector<int64_t> *zero);

  // Generate name2node_dict to save input name and its related nodes
  void UpdateInputNameToNodes();

  void RemoveNodeByName(const std::string &name, const bool &update_io = true);

  void ReplaceInputOfAllNodes(
      const std::string &old_name, const std::string &new_name,
      const std::vector<std::shared_ptr<ONNX_NAMESPACE::NodeProto>>
          &except_nodes = {});

 protected:
  const PaddleParser *parser_;
  OnnxHelper *helper_;
  std::vector<std::shared_ptr<ONNX_NAMESPACE::NodeProto>> *parameters_;
  std::vector<std::shared_ptr<ONNX_NAMESPACE::ValueInfoProto>> *inputs_;
  std::vector<std::shared_ptr<ONNX_NAMESPACE::ValueInfoProto>> *outputs_;
  std::vector<std::shared_ptr<ONNX_NAMESPACE::NodeProto>> *nodes_;
  std::vector<std::string> supported_quantize_type_;
  std::map<std::string, std::vector<std::shared_ptr<ONNX_NAMESPACE::NodeProto>>>
      name2node_dict_;

  virtual void AddQDQ();
};
}  // namespace paddle2onnx
