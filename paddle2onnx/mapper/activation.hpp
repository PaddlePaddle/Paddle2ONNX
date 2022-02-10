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
    op_mapper["softplus"] = "Softplus";
  }

  int32_t GetMinOpset(bool verbose = false) {
    auto op = parser->GetOpDesc(block_idx, op_idx);
    if (op.type() == "softplus") {
      float beta = 0.0;
      float threshold = 20.0;
      parser->GetOpAttr(op, "beta", &beta);
      parser->GetOpAttr(op, "threshold", &threshold);
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
        parser->GetOpInput(block_idx, op_idx, "X");
    std::vector<TensorInfo> output_info =
        parser->GetOpOutput(block_idx, op_idx, "Out");
    auto op = parser->GetOpDesc(block_idx, op_idx);
    auto iter = op_mapper.find(op.type());
    Assert(op_mapper.end() != iter,
           "Cannot find " + op.type() + " in activation op_mapper.");
    helper->MakeNode(iter->second, {input_info[0].name}, {output_info[0].name});
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

  void Opset7(OnnxHelper* helper) {
    std::vector<TensorInfo> input_info =
        parser->GetOpInput(block_idx, op_idx, "X");
    std::vector<TensorInfo> output_info =
        parser->GetOpOutput(block_idx, op_idx, "Out");
    auto op = parser->GetOpDesc(block_idx, op_idx);
    auto node = helper->MakeNode("LeakyRelu", {input_info[0].name},
                                 {output_info[0].name});
    AddAttribute(node, "alpha", alpha);
  }

 private:
  float alpha;
};

class GeluMapper : public Mapper {
 public:
  GeluMapper(const PaddleParser& p, int64_t block_id, int64_t op_id)
      : Mapper(p, block_id, op_id) {}

  int32_t GetMinOpset(bool verbose = false) { return 9; }

  void Opset9(OnnxHelper* helper) {
    std::vector<TensorInfo> input_info =
        parser->GetOpInput(block_idx, op_idx, "X");
    std::vector<TensorInfo> output_info =
        parser->GetOpOutput(block_idx, op_idx, "Out");
    auto input_onnx_dtype = GetOnnxDtype(input_info[0].dtype);
    auto op = parser->GetOpDesc(block_idx, op_idx);
    double sqrt_2_value = 1.4142135623730951;
    double scale_value = 0.5;
    double const_1_value = 1.0;
    auto sqrt_2 = helper->MakeConstant({1}, input_onnx_dtype, sqrt_2_value);
    auto scale = helper->MakeConstant({1}, input_onnx_dtype, scale_value);
    auto const_1 =
        helper->MakeConstant({1}, input_onnx_dtype, const_1_value);

        // the computation formula follows
        // https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/functional/gelu_cn.html#gelu
        auto erf0 =
            helper->MakeNode("Div", {input_info[0].name, sqrt_2->output(0)});
    auto erf1 = helper->MakeNode("Erf", {erf0->output(0)});
    auto gelu0 = helper->MakeNode("Add", {erf1->output(0), const_1->output(0)});
    auto gelu1 =
        helper->MakeNode("Mul", {input_info[0].name, gelu0->output(0)});
    helper->MakeNode("Mul", {gelu1->output(0), scale->output(0)},
                     {output_info[0].name});
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
