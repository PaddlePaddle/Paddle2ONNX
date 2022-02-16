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
#include "paddle2onnx/mapper/activation.h"

namespace paddle2onnx {

REGISTER_MAPPER(relu, ActivationMapper)
REGISTER_MAPPER(relu6, Relu6Mapper)
REGISTER_MAPPER(tanh, ActivationMapper)
REGISTER_MAPPER(log, ActivationMapper)
REGISTER_MAPPER(sigmoid, ActivationMapper)
REGISTER_MAPPER(sqrt, ActivationMapper)
REGISTER_MAPPER(softplus, ActivationMapper)
REGISTER_MAPPER(leaky_relu, LeakyReluMapper)
REGISTER_MAPPER(gelu, GeluMapper)
REGISTER_MAPPER(selu, SeluMapper)
REGISTER_MAPPER(prelu, PReluMapper)
REGISTER_MAPPER(hard_sigmoid, HardSigmoidMapper)
REGISTER_MAPPER(swish, SwishMapper)
REGISTER_MAPPER(hard_swish, HardSwishMapper)
REGISTER_MAPPER(softmax, SoftMaxMapper)

int32_t ActivationMapper::GetMinOpset(bool verbose) {
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
        std::cerr << "Paddle2ONNX only supports softplus with threshold == 20.0"
                  << std::endl;
      }
      return -1;
    }
  }

  return 7;
}

void ActivationMapper::Opset7(OnnxHelper* helper) {
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

void Relu6Mapper::Opset7(OnnxHelper* helper) {
  std::vector<TensorInfo> input_info =
      parser_->GetOpInput(block_idx_, op_idx_, "X");
  std::vector<TensorInfo> output_info =
      parser_->GetOpOutput(block_idx_, op_idx_, "Out");
  auto op = parser_->GetOpDesc(block_idx_, op_idx_);
  float min = 0.0;
  helper->Clip(input_info[0].name, output_info[0].name, min, threshold_,
               input_info[0].dtype);
}

void PReluMapper::Opset7(OnnxHelper* helper) {
  std::vector<TensorInfo> input_info =
      parser_->GetOpInput(block_idx_, op_idx_, "X");
  std::vector<TensorInfo> slope_info =
      parser_->GetOpInput(block_idx_, op_idx_, "Alpha");
  std::vector<TensorInfo> output_info =
      parser_->GetOpOutput(block_idx_, op_idx_, "Out");

  auto op = parser_->GetOpDesc(block_idx_, op_idx_);

  std::string slope_cast_name = slope_info[0].name;
  if (slope_info[0].dtype == P2ODataType::FP64) {
    slope_cast_name = helper->AutoCast({slope_info[0].name}, P2ODataType::FP64,
                                       P2ODataType::FP32);
  }

  if (input_info[0].dtype == P2ODataType::FP64) {
    std::string x_cast_name = helper->AutoCast(
        {input_info[0].name}, P2ODataType::FP64, P2ODataType::FP32);
    auto node = helper->MakeNode("PRelu", {x_cast_name, slope_cast_name});
    helper->AutoCast(node->output(0), {output_info[0].name}, P2ODataType::FP32,
                     P2ODataType::FP64);
  } else {
    helper->MakeNode("PRelu", {input_info[0].name, slope_cast_name},
                     {output_info[0].name});
  }
}

void SeluMapper::Opset7(OnnxHelper* helper) {
  std::vector<TensorInfo> input_info =
      parser_->GetOpInput(block_idx_, op_idx_, "X");
  std::vector<TensorInfo> output_info =
      parser_->GetOpOutput(block_idx_, op_idx_, "Out");
  auto op = parser_->GetOpDesc(block_idx_, op_idx_);
  auto node =
      helper->MakeNode("Selu", {input_info[0].name}, {output_info[0].name});
  AddAttribute(node, "alpha", alpha_);
  AddAttribute(node, "gamma", scale_);
}

void HardSigmoidMapper::Opset7(OnnxHelper* helper) {
  std::vector<TensorInfo> input_info =
      parser_->GetOpInput(block_idx_, op_idx_, "X");
  std::vector<TensorInfo> output_info =
      parser_->GetOpOutput(block_idx_, op_idx_, "Out");
  auto op = parser_->GetOpDesc(block_idx_, op_idx_);
  auto node = helper->MakeNode("HardSigmoid", {input_info[0].name},
                               {output_info[0].name});
  AddAttribute(node, "alpha", alpha_);
  AddAttribute(node, "beta", beta_);
}

void SwishMapper::Opset7(OnnxHelper* helper) {
  std::vector<TensorInfo> input_info =
      parser_->GetOpInput(block_idx_, op_idx_, "X");
  std::vector<TensorInfo> output_info =
      parser_->GetOpOutput(block_idx_, op_idx_, "Out");
  auto op = parser_->GetOpDesc(block_idx_, op_idx_);

  std::string beta_node =
      helper->MakeConstant({1}, GetOnnxDtype(input_info[0].dtype), beta_)
          ->output(0);
  auto beta_x_node = helper->MakeNode("Mul", {input_info[0].name, beta_node});
  auto sigmod_node = helper->MakeNode("Sigmoid", {beta_x_node->output(0)});
  helper->MakeNode("Mul", {input_info[0].name, sigmod_node->output(0)},
                   {output_info[0].name});
}

void HardSwishMapper::Opset7(OnnxHelper* helper) {
  std::vector<TensorInfo> input_info =
      parser_->GetOpInput(block_idx_, op_idx_, "X");
  std::vector<TensorInfo> output_info =
      parser_->GetOpOutput(block_idx_, op_idx_, "Out");
  auto op = parser_->GetOpDesc(block_idx_, op_idx_);

  std::string scale_node =
      helper->MakeConstant({1}, GetOnnxDtype(input_info[0].dtype), scale_)
          ->output(0);
  std::string offset_node =
      helper->MakeConstant({1}, GetOnnxDtype(input_info[0].dtype), offset_)
          ->output(0);

  auto add_node = helper->MakeNode("Add", {input_info[0].name, offset_node});
  auto clip_node =
      helper->Clip(add_node->output(0), 0.0, threshold_, input_info[0].dtype);

  auto mul_node = helper->MakeNode("Mul", {input_info[0].name, clip_node});
  helper->MakeNode("Div", {mul_node->output(0), scale_node},
                   {output_info[0].name});
}

void LeakyReluMapper::Opset7(OnnxHelper* helper) {
  std::vector<TensorInfo> input_info =
      parser_->GetOpInput(block_idx_, op_idx_, "X");
  std::vector<TensorInfo> output_info =
      parser_->GetOpOutput(block_idx_, op_idx_, "Out");
  auto op = parser_->GetOpDesc(block_idx_, op_idx_);
  auto node = helper->MakeNode("LeakyRelu", {input_info[0].name},
                               {output_info[0].name});
  AddAttribute(node, "alpha", alpha_);
}

void GeluMapper::Opset9(OnnxHelper* helper) {
  std::vector<TensorInfo> input_info =
      parser_->GetOpInput(block_idx_, op_idx_, "X");
  std::vector<TensorInfo> output_info =
      parser_->GetOpOutput(block_idx_, op_idx_, "Out");
  auto input_onnx_dtype = GetOnnxDtype(input_info[0].dtype);
  auto op = parser_->GetOpDesc(block_idx_, op_idx_);
  double sqrt_2_value = 1.4142135623730951;
  double scale_value = 0.5;
  double const_1_value = 1.0;
  auto sqrt_2 = helper->MakeConstant({1}, ONNX_NAMESPACE::TensorProto::FLOAT,
                                     sqrt_2_value);
  auto scale = helper->MakeConstant({1}, ONNX_NAMESPACE::TensorProto::FLOAT,
                                    scale_value);
  auto const_1 = helper->MakeConstant({1}, ONNX_NAMESPACE::TensorProto::FLOAT,
                                      const_1_value);

  auto input_name = helper->AutoCast(input_info[0].name, input_info[0].dtype,
                                     P2ODataType::FP32);

  // the computation formula follows
  // https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/functional/gelu_cn.html#gelu
  auto erf0 = helper->MakeNode("Div", {input_name, sqrt_2->output(0)});
  auto erf1 = helper->MakeNode("Erf", {erf0->output(0)});
  auto gelu0 = helper->MakeNode("Add", {erf1->output(0), const_1->output(0)});
  auto gelu1 = helper->MakeNode("Mul", {input_name, gelu0->output(0)});

  if (input_info[0].dtype != P2ODataType::FP32) {
    auto out = helper->MakeNode("Mul", {gelu1->output(0), scale->output(0)});
    auto cast_out =
        helper->MakeNode("Cast", {out->output(0)}, {output_info[0].name});
    AddAttribute(cast_out, "to", GetOnnxDtype(input_info[0].dtype));
  } else {
    helper->MakeNode("Mul", {gelu1->output(0), scale->output(0)},
                     {output_info[0].name});
  }
}

void SoftMaxMapper::Opset7(OnnxHelper* helper) {
  auto op = parser_->GetOpDesc(block_idx_, op_idx_);
  std::vector<TensorInfo> input_info =
      parser_->GetOpInput(block_idx_, op_idx_, "X");
  std::vector<TensorInfo> output_info =
      parser_->GetOpOutput(block_idx_, op_idx_, "Out");
  if (axis_ < 0) {
    axis_ = axis_ + output_info[0].Rank();
  }
  if (axis_ == output_info[0].Rank() - 1) {
    auto node = helper->MakeNode("Softmax", {input_info[0].name},
                                 {output_info[0].name});
    AddAttribute(node, "axis", axis_);
  } else {
    std::vector<int64_t> perm = Arange(0, output_info[0].Rank());
    perm[output_info[0].Rank() - 1] = axis_;
    perm[axis_] = output_info[0].Rank() - 1;
    auto transpose_node = helper->MakeNode("Transpose", {input_info[0].name});
    AddAttribute(transpose_node, "perm", perm);
    auto softmax_node =
        helper->MakeNode("Softmax", {transpose_node->output(0)});
    int64_t axis_last = -1;
    AddAttribute(softmax_node, "axis", axis_last);
    auto transpose_node_last = helper->MakeNode(
        "Transpose", {softmax_node->output(0)}, {output_info[0].name});
    AddAttribute(transpose_node_last, "perm", perm);
  }
}

void SoftMaxMapper::Opset13(OnnxHelper* helper) {
  auto op = parser_->GetOpDesc(block_idx_, op_idx_);
  int64_t axis;
  parser_->GetOpAttr(op, "axis", &axis);
  std::vector<TensorInfo> input_info =
      parser_->GetOpInput(block_idx_, op_idx_, "X");
  std::vector<TensorInfo> output_info =
      parser_->GetOpOutput(block_idx_, op_idx_, "Out");
  auto node =
      helper->MakeNode("Softmax", {input_info[0].name}, {output_info[0].name});
  AddAttribute(node, "axis", axis);
}

}  // namespace paddle2onnx
