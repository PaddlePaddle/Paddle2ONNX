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
#include "paddle2onnx/mapper/mapper.hpp"

namespace paddle2onnx {

class ActivationMapper : public Mapper {
 public:
  ActivationMapper(const PaddleParser& p, int64_t block_id, int64_t op_id)
      : Mapper(p, block_id, op_id) {
    op_mapper["relu"] = "Relu";
    op_mapper["tanh"] = "Tanh";
    op_mapper["log"] = "Log";
    op_mapper["sigmoid"] = "Sigmoid";
    op_mapper["sqrt"] = "Sqrt";
    op_mapper["softplu"] = "Softplus";
  }

  int32_t GetMinOpset(bool verbose = false) {
    auto op = parser->GetOpDesc(block_idx, op_idx);
    if (op.type() == "softplus") {
      float beta = 0.0;
      float threshold = 20.0;
      parser->GetOpAttr(op, "beta", &beta);
      parser->GetOpAttr(op, "threshold", &threshold);
      if (beta > 1e-06 || beta < -1e-06) {
        if (verbose) {
          std::cerr << "Paddle2ONNX only supports softplus with beta == 0"
                    << std::endl;
        }
        return -1;
      }
      if ((threshold - 20.0) > 1e-06 || (threshold - 20.0) < -1e-06) {
        if (verbose) {
          std::cerr
              << "Paddle2ONNX only supports softplus with threshold == 20.0"
              << std::endl;
        }
        return -1;
      }
    }

    return 7;
  }

  void Opset7(std::vector<std::shared_ptr<ONNX_NAMESPACE::NodeProto>>* nodes) {
    nodes->clear();
    std::vector<TensorInfo> input_info =
        parser->GetOpInput(block_idx, op_idx, "X");
    std::vector<TensorInfo> output_info =
        parser->GetOpOutput(block_idx, op_idx, "Out");
    auto op = parser->GetOpDesc(block_idx, op_idx);
    auto iter = op_mapper.find(op.type());
    Assert(op_mapper.end() != iter,
           "Cannot find " + op.type() + " in activation op_mapper.");
    auto node =
        MakeNode(iter->second, {input_info[0].name}, {output_info[0].name});
    nodes->push_back(node);
  }

 private:
  std::map<std::string, std::string> op_mapper;
};

class LeakyReluMapper : public Mapper {
 public:
  LeakyReluMapper(const PaddleParser& p, int64_t block_id, int64_t op_id)
      : Mapper(p, block_id, op_id) {
    auto op = parser->GetOpDesc(block_idx, op_idx);
    parser->GetOpAttr(op, "alpha", &alpha);
  }

  int32_t GetMinOpset(bool verbose = false) { return 7; }

  void Opset7(std::vector<std::shared_ptr<ONNX_NAMESPACE::NodeProto>>* nodes) {
    nodes->clear();
    std::vector<TensorInfo> input_info =
        parser->GetOpInput(block_idx, op_idx, "X");
    std::vector<TensorInfo> output_info =
        parser->GetOpOutput(block_idx, op_idx, "Out");
    auto op = parser->GetOpDesc(block_idx, op_idx);
    auto node =
        MakeNode("LeakyRelu", {input_info[0].name}, {output_info[0].name});
    AddAttribute(node, "alpha", alpha);
    nodes->push_back(node);
  }

 private:
  float alpha;
};

class PReluMapper : public Mapper {
 public:
  PReluMapper(const PaddleParser& p, int64_t block_id, int64_t op_id)
      : Mapper(p, block_id, op_id) {}

  int32_t GetMinOpset(bool verbose = false) { return 7; }

  void Opset7(std::vector<std::shared_ptr<ONNX_NAMESPACE::NodeProto>>* nodes) {
    nodes->clear();
    std::vector<TensorInfo> input_info =
        parser->GetOpInput(block_idx, op_idx, "X");
    std::vector<TensorInfo> slope_info =
        parser->GetOpInput(block_idx, op_idx, "Alpha");
    std::vector<TensorInfo> output_info =
        parser->GetOpOutput(block_idx, op_idx, "Out");

    auto op = parser->GetOpDesc(block_idx, op_idx);
    std::string x_name = input_info[0].name;
    std::string slope_name = slope_info[0].name;
    if (input_info[0].dtype != P2ODataType::FP32) {
      x_name = MapperHelper::Get()->GenName("prelu.cast");
      auto cast_node = MakeNode("Cast", {input_info[0].name}, {x_name});
      int64_t x = 5;
      AddAttribute(cast_node, "to", x);
      nodes->push_back(cast_node);
    }
    if (slope_info[0].dtype != P2ODataType::FP32) {
      slope_name = MapperHelper::Get()->GenName("prelu.cast");
      auto cast_node = MakeNode("Cast", {slope_info[0].name}, {slope_name});
      AddAttribute(cast_node, "to", ONNX_NAMESPACE::TensorProto::FLOAT);
      nodes->push_back(cast_node);
    }
    if (output_info[0].dtype != P2ODataType::FP32) {
      std::string out_name = MapperHelper::Get()->GenName("prelu.before_cast");
      auto node = MakeNode("PRelu", {x_name, slope_name}, {out_name});
      auto cast_node = MakeNode("Cast", {out_name}, {output_info[0].name});
      AddAttribute(cast_node, "to", GetOnnxDtype(output_info[0].dtype));
      nodes->push_back(node);
      nodes->push_back(cast_node);
    } else {
      auto node =
          MakeNode("PRelu", {x_name, slope_name}, {output_info[0].name});
      nodes->push_back(node);
    }
  }
};

REGISTER_MAPPER(relu, ActivationMapper)
REGISTER_MAPPER(tanh, ActivationMapper)
REGISTER_MAPPER(log, ActivationMapper)
REGISTER_MAPPER(sigmoid, ActivationMapper)
REGISTER_MAPPER(sqrt, ActivationMapper)
REGISTER_MAPPER(softplus, ActivationMapper)
REGISTER_MAPPER(leaky_relu, LeakyReluMapper)
REGISTER_MAPPER(prelu, PReluMapper)
REGISTER_MAPPER(hard_sigmoid, PReluMapper)
}  // namespace paddle2onnx
