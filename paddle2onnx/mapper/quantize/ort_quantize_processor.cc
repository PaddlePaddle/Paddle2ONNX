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

#include "paddle2onnx/mapper/quantize/ort_quantize_processor.h"

namespace paddle2onnx {
ORTQuantizeProcessor::ORTQuantizeProcessor() {
  supported_quantize_type_ = {
      "Add",
      "Conv",
      "LeakyRelu"
      "MatMul",
      "Mul",
      "Relu",
      "Sigmoid",
  };
}

// According to:
// https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/core/optimizer/qdq_transformer/selectors_actions/qdq_selector_action_transformer.cc
void ORTQuantizeProcessor::AddQDQ() {
  BaseQuantizeProcessor::AddQDQ();
  for (auto iter = nodes_->begin(); iter < nodes_->end(); iter++) {
    auto node = *iter;
    auto type_iter = std::find(supported_quantize_type_.begin(),
                               supported_quantize_type_.end(), node->op_type());
    if (!supported_quantize_type_.empty() &&
        type_iter == supported_quantize_type_.end()) {
      continue;
    }
    if (node->op_type() == "MatMul") {
      std::vector<std::string> tensor_names = {node->input(0), node->input(1),
                                               node->output(0)};
      for (auto& name : tensor_names) {
        if (helper_->quantize_info.find(name) != helper_->quantize_info.end()) {
          continue;
        }
        std::vector<float> matmul_weight;
        if (!GetTensorByName(name, &matmul_weight)) {
          continue;
        }
        std::vector<int64_t> matmul_weight_shape;
        if (!GetTensorShape(name, &matmul_weight_shape)) {
          continue;
        }
        int64_t quantize_axis = 1;
        std::vector<float> scale;
        std::vector<int64_t> zeros;
        GetChannelWiseQuantizeInfo(matmul_weight, matmul_weight_shape,
                                   quantize_axis, &scale, &zeros);
        auto scale_node =
            helper_->Constant(ONNX_NAMESPACE::TensorProto::FLOAT, scale);
        auto zero_node =
            helper_->Constant(ONNX_NAMESPACE::TensorProto::INT8, zeros);
        QuantizeInfo matmul_weight_quantize_info(scale, zeros, scale_node,
                                                 zero_node, quantize_axis);
        helper_->quantize_info[name] = matmul_weight_quantize_info;
      }
      if (!CanBeQuantize(tensor_names)) {
        tensor_names.pop_back();
        if (!CanBeQuantize(tensor_names)) {
          continue;
        }
      }
      for (auto& name : tensor_names) {
        AppendQuantizeTensor(name);
      }
    }

    std::vector<std::string> tensor_names;
    for (size_t i = 0; i < node->input_size(); ++i) {
      std::string node_input = node->input(i);
      tensor_names.push_back(node_input);
    }
    for (size_t i = 0; i < node->output_size(); ++i) {
      std::string node_output = node->output(i);
      tensor_names.push_back(node_output);
    }
    if (!CanBeQuantize(tensor_names)) {
      continue;
    }
    for (auto& name : tensor_names) {
      AppendQuantizeTensor(name);
    }
  }
}

void ORTQuantizeProcessor::ProcessQuantizeModel(
    std::vector<std::shared_ptr<ONNX_NAMESPACE::NodeProto>>* parameters,
    std::vector<std::shared_ptr<ONNX_NAMESPACE::ValueInfoProto>>* inputs,
    std::vector<std::shared_ptr<ONNX_NAMESPACE::ValueInfoProto>>* outputs,
    std::vector<std::shared_ptr<ONNX_NAMESPACE::NodeProto>>* nodes,
    OnnxHelper* helper, const PaddleParser& parser,
    std::string* calibration_cache) {
  BaseQuantizeProcessor::ProcessQuantizeModel(
      parameters, inputs, outputs, nodes, helper, parser, calibration_cache);

  // When deploy_backend is ONNXRuntime, use the follow four steps to process:
  // 1. broadcast quantize info
  // 2. remove all quantize ops
  // 3. merge conv and add
  // 4. merge conv and bn
  // 5. add Q and DQ according ONNXRuntime quantize OP fuse patten.
  // 6. use topo sort in nodes
  QuantizeInfoBroadcast();
  RemoveAllQuantizeOps();
  MergeConvAdd();
  MergeConvBN();
  AddQDQ();
  UpdateInputNameToNodes();
  AddQDQInModel();
  SortNodes();
}
}  // namespace paddle2onnx