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
    op_mapper_["relu"] = "Relu";
    op_mapper_["tanh"] = "Tanh";
    op_mapper_["log"] = "Log";
    op_mapper_["sigmoid"] = "Sigmoid";
    op_mapper_["sqrt"] = "Sqrt";
    op_mapper_["softplus"] = "Softplus";
  }

  int32_t GetMinOpset(bool verbose = false) {
    auto op = parser_->GetOpDesc(block_idx_, op_idx_);
    if (op.type() == "softplus") {
      float beta = 0.0;
      float threshold = 20.0;
      parser_->GetOpAttr(op, "beta", &beta);
      parser_->GetOpAttr(op, "threshold", &threshold);
      if ((beta - 1.0) > 1e-06 || (beta - 1.0) < -1e-06) {
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

  void Opset7(OnnxHelper* helper) {
    std::vector<TensorInfo> input_info =
        parser_->GetOpInput(block_idx_, op_idx_, "X");
    std::vector<TensorInfo> output_info =
        parser_->GetOpOutput(block_idx_, op_idx_, "Out");
    auto op = parser_->GetOpDesc(block_idx_, op_idx_);
    auto iter = op_mapper_.find(op.type());
    Assert(op_mapper_.end() != iter,
           "Cannot find " + op.type() + " in activation op_mapper.");
    helper->MakeNode(iter->second, {input_info[0].name}, {output_info[0].name});
  }

 private:
  std::map<std::string, std::string> op_mapper_;
};

class LeakyReluMapper : public Mapper {
 public:
  LeakyReluMapper(const PaddleParser& p, int64_t block_id, int64_t op_id)
      : Mapper(p, block_id, op_id) {
    auto op = parser_->GetOpDesc(block_idx_, op_idx_);
    parser_->GetOpAttr(op, "alpha", &alpha_);
  }

  int32_t GetMinOpset(bool verbose = false) { return 7; }

  void Opset7(OnnxHelper* helper) {
    std::vector<TensorInfo> input_info =
        parser_->GetOpInput(block_idx_, op_idx_, "X");
    std::vector<TensorInfo> output_info =
        parser_->GetOpOutput(block_idx_, op_idx_, "Out");
    auto op = parser_->GetOpDesc(block_idx_, op_idx_);
    auto node = helper->MakeNode("LeakyRelu", {input_info[0].name},
                                 {output_info[0].name});
    AddAttribute(node, "alpha", alpha_);
  }

 private:
  float alpha_;
};

class GeluMapper : public Mapper {
 public:
  GeluMapper(const PaddleParser& p, int64_t block_id, int64_t op_id)
      : Mapper(p, block_id, op_id) {}

  int32_t GetMinOpset(bool verbose = false) { return 9; }

  void Opset9(OnnxHelper* helper) {
    std::vector<TensorInfo> input_info =
        parser_->GetOpInput(block_idx_, op_idx_, "X");
    std::vector<TensorInfo> output_info =
        parser_->GetOpOutput(block_idx_, op_idx_, "Out");
    auto input_onnx_dtype = GetOnnxDtype(input_info[0].dtype);
    auto op = parser_->GetOpDesc(block_idx_, op_idx_);
    double sqrt_2_value = 1.4142135623730951;
    double scale_value = 0.5;
    double const_1_value = 1.0;
    auto sqrt_2 = helper->MakeConstant({1}, ONNX_NAMESPACE::TensorProto::FLOAT, sqrt_2_value);
    auto scale = helper->MakeConstant({1}, ONNX_NAMESPACE::TensorProto::FLOAT, scale_value);
    auto const_1 = helper->MakeConstant({1}, ONNX_NAMESPACE::TensorProto::FLOAT, const_1_value);
 
    auto input_name = helper->AutoCast(input_info[0].name, input_info[0].dtype, P2ODataType::FP32);

    // the computation formula follows
    // https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/functional/gelu_cn.html#gelu
    auto erf0 =
        helper->MakeNode("Div", {input_name, sqrt_2->output(0)});
    auto erf1 = helper->MakeNode("Erf", {erf0->output(0)});
    auto gelu0 = helper->MakeNode("Add", {erf1->output(0), const_1->output(0)});
    auto gelu1 =
        helper->MakeNode("Mul", {input_name, gelu0->output(0)});
    
    if (input_info[0].dtype != P2ODataType::FP32) {
      auto out = helper->MakeNode("Mul", {gelu1->output(0), scale->output(0)});
      auto cast_out = helper->MakeNode("Cast", {out->output(0)}, {output_info[0].name});
      AddAttribute(cast_out, "to", GetOnnxDtype(input_info[0].dtype));
    } else {
      helper->MakeNode("Mul", {gelu1->output(0), scale->output(0)},
                      {output_info[0].name});
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
REGISTER_MAPPER(gelu, GeluMapper)
// REGISTER_MAPPER(prelu, PReluMapper)
}  // namespace paddle2onnx
