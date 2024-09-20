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

#include "paddle2onnx/mapper/quantize/tensorrt_quantize_processor.h"

namespace paddle2onnx {
// In TensorRT, all quantized op: Conv, ConvTranspose, liner(MatMul), MaxPool,
// AvgPool, AdaptiveAvgPool, rnn(not support now)
// According to:
// https://github.com/NVIDIA/TensorRT/tree/main/tools/pytorch-quantization/pytorch_quantization/nn/modules
void TensorRTQuantizeProcessor::AddQDQ() {
  BaseQuantizeProcessor::AddQDQ();
  std::vector<std::string>
      quantize_tensors;  // save the tensor names that need add quantize ops
  std::vector<std::string> pool_types = {"MaxPool", "AvgPool",
                                         "AdaptiveAvgPool"};
  for (auto iter = nodes_->begin(); iter < nodes_->end(); iter++) {
    quantize_tensors.clear();
    auto node = *iter;
    if (node->op_type() == "Conv" || node->op_type() == "ConvTranspose") {
      std::vector<std::string> tensor_names = {node->input(0), node->input(1)};
      if (!CanBeQuantize(tensor_names)) {
        continue;
      }
      quantize_tensors = tensor_names;
    }
    if (node->op_type() == "MatMul") {
      std::vector<std::string> tensor_names = {node->input(0), node->input(1)};
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
        continue;
      }
      quantize_tensors = tensor_names;
    }
    auto type_iter =
        std::find(pool_types.begin(), pool_types.end(), node->op_type());
    if (type_iter != pool_types.end()) {
      std::vector<std::string> tensor_names = {node->input(0)};
      if (!CanBeQuantize(tensor_names)) {
        continue;
      }
      quantize_tensors = tensor_names;
    }

    std::string negative_scale_tensor = "";
    for (std::string& name : quantize_tensors) {
      Assert(helper_->quantize_info.find(name) != helper_->quantize_info.end(),
             "[BaseQuantizeProcessor] Can not find quantize info for tensor: " +
                 name);
      QuantizeInfo quantize_info = helper_->quantize_info[name];
      std::vector<float> scales = quantize_info.scale_;
      for (auto& i : scales) {
        if (i <= 1e-10) {
          negative_scale_tensor = negative_scale_tensor + " " + name;
        }
      }
    }
    if (negative_scale_tensor.size() > 0) {
      P2OLogger()
          << "[Warning] The scale of tensors: [ " + negative_scale_tensor +
                 " ] contains negative scale, so this OP will not be quantized."
          << std::endl;
      continue;
    }
    // An OP requires a separate quantize op
    for (std::string& name : quantize_tensors) {
      if (IsGraphOutput(name)) {
        continue;
      }
      QuantizeInfo quantize_info = helper_->quantize_info[name];
      std::string scale_node = quantize_info.scale_node_;
      std::string zeros_node = quantize_info.zeros_node_;
      int64_t quantize_axis = quantize_info.quantize_axis_;
      auto q_node =
          helper_->MakeNode("QuantizeLinear", {name, scale_node, zeros_node});
      if (helper_->GetOpsetVersion() >= 13) {
        AddAttribute(q_node, "axis", quantize_axis);
      }
      auto dq_node = helper_->MakeNode(
          "DequantizeLinear", {q_node->output(0), scale_node, zeros_node});
      if (helper_->GetOpsetVersion() >= 13) {
        AddAttribute(dq_node, "axis", quantize_axis);
      }
      for (size_t i = 0; i < node->input_size(); ++i) {
        if (node->input(i) == name) {
          node->set_input(i, dq_node->output(0));
        }
      }
    }
  }
}

void TensorRTQuantizeProcessor::GenerateCache(std::string* calibration_cache) {
  union {
    float f;
    unsigned char farray[4];
  } un;
  *calibration_cache += "TRT-8XXX-EntropyCalibration2 \n";
  for (auto iter = helper_->quantize_info.rbegin();
       iter != helper_->quantize_info.rend(); iter++) {
    std::string tensor_name = iter->first;
    QuantizeInfo quantize_info = iter->second;
    if (quantize_info.scale_.size() == 1) {
      float val = quantize_info.scale_[0];
      un.f = val;
      *calibration_cache += (tensor_name + ": ");
      std::stringstream enc;
      for (int64_t i = 3; i >= 0; i--) {
        enc << std::hex << std::setw(2) << std::setfill('0')
            << (int)(un.farray[i]);
      }
      *calibration_cache = *calibration_cache + enc.str() + "\n";
    }
  }
}

void TensorRTQuantizeProcessor::ProcessQuantizeModel(
    std::vector<std::shared_ptr<ONNX_NAMESPACE::NodeProto>>* parameters,
    std::vector<std::shared_ptr<ONNX_NAMESPACE::ValueInfoProto>>* inputs,
    std::vector<std::shared_ptr<ONNX_NAMESPACE::ValueInfoProto>>* outputs,
    std::vector<std::shared_ptr<ONNX_NAMESPACE::NodeProto>>* nodes,
    OnnxHelper* helper, const PaddleParser& parser,
    std::string* calibration_cache) {
  BaseQuantizeProcessor::ProcessQuantizeModel(
      parameters, inputs, outputs, nodes, helper, parser, calibration_cache);

  // When deploy_backend is TensorRT, use the follow four steps to process:
  // For Explicit Quantization
  // 1. broadcast quantize info
  // 2. remove all quantize ops
  // 3. add Q and DQ before conv and matmul.
  // 4. use topo sort in nodes

  // For Implicit Quantization
  // 1. remove all quantize ops
  // 2. broadcast quantize info
  // 3. save float onnx model and alibration.cache
  QuantizeInfoBroadcast();
  RemoveAllQuantizeOps();
  // Add qdq for Explicit Quantization
  // AddTrtQDQ();
  // SortNodes();

  // Genarate calibration.cache for Implicit Quantization
  // convert float to hex
  GenerateCache(calibration_cache);
}
}  // namespace paddle2onnx