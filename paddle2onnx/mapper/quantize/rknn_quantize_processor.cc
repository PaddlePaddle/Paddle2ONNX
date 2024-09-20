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

#include "paddle2onnx/mapper/quantize/rknn_quantize_processor.h"

namespace paddle2onnx {
RKNNQuantizeProcessor::RKNNQuantizeProcessor() {
  supported_quantize_type_ = {"Abs",
                              "Acos",
                              "Add",
                              "Asin",
                              "Atan",
                              "AveragePool",
                              "BatchNormalization",
                              "Ceil",
                              "Clip",
                              "Conv",
                              "ConvTranspose",
                              "Cos",
                              "Cosh",
                              "Concat",
                              "Div",
                              "Elu",
                              "Erf",
                              "Exp",
                              "Floor",
                              "Gemm",
                              "GlobalAveragePool",
                              "HardSigmoid",
                              "HardSwish",
                              "InstanceNormalization",
                              "IsInf",
                              "IsNaN",
                              "Log",
                              "MatMul",
                              "MaxPool",
                              "Mul",
                              "Neg",
                              "ReduceMean",
                              "Relu",
                              "Reshape",
                              "Resize",
                              "Round",
                              "Shape",
                              "Sigmoid",
                              "Sin",
                              "Sinh",
                              "Slice",
                              "Softmax",
                              "Split",
                              "Sqrt",
                              "Tan",
                              "Tanh",
                              "Transpose"};
}

void RKNNQuantizeProcessor::AddQDQ() {
  BaseQuantizeProcessor::AddQDQ();
  for (auto iter = nodes_->begin(); iter < nodes_->end(); iter++) {
    auto node = *iter;
    auto type_iter = std::find(supported_quantize_type_.begin(),
                               supported_quantize_type_.end(), node->op_type());
    if (type_iter == supported_quantize_type_.end()) {
      continue;
    }

    std::vector<std::string> tensor_names = {};
    for (size_t i = 0; i < node->input_size(); ++i) {
      std::string node_input = node->input(i);
      tensor_names.push_back(node_input);
    }
    for (size_t i = 0; i < node->output_size(); ++i) {
      std::string node_output = node->output(i);
      tensor_names.push_back(node_output);
    }

    if (node->op_type() == "MatMul" || node->op_type() == "Mul") {
      for (auto& name : tensor_names) {
        if (helper_->quantize_info.find(name) != helper_->quantize_info.end()) {
          continue;
        }
        std::vector<float> weight_data;
        if (!GetTensorByName(name, &weight_data)) {
          continue;
        }
        std::vector<int64_t> weight_shape;
        if (!GetTensorShape(name, &weight_shape)) {
          continue;
        }

        int64_t quantize_axis = 1;
        std::vector<float> scale;
        std::vector<int64_t> zeros;
        std::string scale_node, zero_node;
        if (weight_shape.size() <= 1) {
          GetTensorWiseQuantizeInfo(weight_data, &scale, &zeros);
          scale_node = helper_->Constant({}, ONNX_NAMESPACE::TensorProto::FLOAT,
                                         scale[0]);
          zero_node = helper_->Constant({}, ONNX_NAMESPACE::TensorProto::INT8,
                                        zeros[0]);
        } else {
          GetChannelWiseQuantizeInfo(weight_data, weight_shape, quantize_axis,
                                     &scale, &zeros);
          scale_node =
              helper_->Constant(ONNX_NAMESPACE::TensorProto::FLOAT, scale);
          zero_node =
              helper_->Constant(ONNX_NAMESPACE::TensorProto::INT8, zeros);
        }
        QuantizeInfo matmul_weight_quantize_info(scale, zeros, scale_node,
                                                 zero_node, quantize_axis);
        helper_->quantize_info[name] = matmul_weight_quantize_info;
      }
    } else if (node->op_type() == "Add") {
      for (auto& name : tensor_names) {
        if (helper_->quantize_info.find(name) != helper_->quantize_info.end()) {
          continue;
        }

        std::vector<float> weight;
        if (!GetTensorByName(name, &weight)) {
          P2OLogger() << "Failed to GetTensorByName: " << node->name() << ";"
                      << name << std::endl;
          continue;
        }

        int64_t quantize_axis = 1;
        std::vector<float> scale;
        std::vector<int64_t> zeros;
        GetTensorWiseQuantizeInfo(weight, &scale, &zeros);

        std::string scale_node, zero_node;
        scale_node =
            helper_->Constant({}, ONNX_NAMESPACE::TensorProto::FLOAT, scale[0]);
        zero_node =
            helper_->Constant({}, ONNX_NAMESPACE::TensorProto::INT8, zeros[0]);

        QuantizeInfo quantize_info(scale, zeros, scale_node, zero_node,
                                   quantize_axis);
        helper_->quantize_info[name] = quantize_info;
      }
    } else if (node->op_type() == "BatchNormalization") {
      // BatchNormalization only need quntize X and Y.
      // when opset > 9, tensor_names is {X, scale, B, input_mean, input_var, Y,
      // running_mean, running_var} when opset <= 9, tensor_names is {X, scale,
      // B, mean, var, Y, mean, var, saved_mean, saved_var}
      tensor_names.erase(tensor_names.begin() + 1, tensor_names.begin() + 5);
      tensor_names.erase(tensor_names.begin() + 2, tensor_names.end());
    }

    if (!CanBeQuantize(tensor_names)) {
      continue;
    }

    for (auto& name : tensor_names) {
      AppendQuantizeTensor(name);
    }
  }
}

void RKNNQuantizeProcessor::PerchannelToPerlayer() {
  UpdateInputNameToNodes();
  for (auto iter = nodes_->begin(); iter < nodes_->end(); iter++) {
    auto node = *iter;

    if (node->op_type() != "MatMul" && node->op_type() != "Mul") {
      continue;
    }

    auto next_nodes = name2node_dict_[node->output(0)];
    if (next_nodes.size() > 1 || IsGraphOutput(node->output(0))) {
      continue;
    }

    auto add_node = next_nodes[0];
    if (add_node->op_type() != "Add" || IsGraphOutput(add_node->output(0))) {
      continue;
    }

    std::vector<std::string> tensor_names = {};
    for (size_t i = 0; i < node->input_size(); ++i) {
      std::string node_input = node->input(i);
      tensor_names.push_back(node_input);
    }
    for (size_t i = 0; i < node->output_size(); ++i) {
      std::string node_output = node->output(i);
      tensor_names.push_back(node_output);
    }

    for (auto& name : tensor_names) {
      if (helper_->quantize_info.find(name) == helper_->quantize_info.end()) {
        continue;
      }

      auto ori_quantize_info = helper_->quantize_info[name];
      auto ori_scale = ori_quantize_info.scale_;

      int64_t now_quantize_axis = 1;
      std::vector<float> now_scale = {
          *std::max_element(ori_scale.begin(), ori_scale.end())};
      std::vector<int64_t> now_zeros = {0};

      std::string scale_node, zero_node;
      scale_node = helper_->Constant({}, ONNX_NAMESPACE::TensorProto::FLOAT,
                                     now_scale[0]);
      zero_node = helper_->Constant({}, ONNX_NAMESPACE::TensorProto::INT8,
                                    now_zeros[0]);

      QuantizeInfo now_quantize_info(now_scale, now_zeros, scale_node,
                                     zero_node, now_quantize_axis);
      helper_->quantize_info[name] = now_quantize_info;
    }
  }
}

void RKNNQuantizeProcessor::ProcessQuantizeModel(
    std::vector<std::shared_ptr<ONNX_NAMESPACE::NodeProto>>* parameters,
    std::vector<std::shared_ptr<ONNX_NAMESPACE::ValueInfoProto>>* inputs,
    std::vector<std::shared_ptr<ONNX_NAMESPACE::ValueInfoProto>>* outputs,
    std::vector<std::shared_ptr<ONNX_NAMESPACE::NodeProto>>* nodes,
    OnnxHelper* helper, const PaddleParser& parser,
    std::string* calibration_cache) {
  BaseQuantizeProcessor::ProcessQuantizeModel(
      parameters, inputs, outputs, nodes, helper, parser, calibration_cache);

  // When deploy_backend is RKNN, use the follow four steps to process:
  // 1. broadcast quantize info
  // 2. remove all quantize ops
  // 3. add Q and DQ
  // 4. use topo sort in nodes
  QuantizeInfoBroadcast();
  RemoveAllQuantizeOps();
  RemoveIdentityOp();
  // MergeConvAdd();
  // MergeConvBN();
  AddQDQ();
  PerchannelToPerlayer();
  UpdateInputNameToNodes();
  AddQDQInModel();
  SortNodes();
}
}  // namespace paddle2onnx