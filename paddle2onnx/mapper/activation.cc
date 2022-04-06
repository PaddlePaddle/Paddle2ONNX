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
REGISTER_MAPPER(exp, ActivationMapper)
REGISTER_MAPPER(floor, ActivationMapper)
REGISTER_MAPPER(cos, ActivationMapper)
REGISTER_MAPPER(sin, ActivationMapper)
REGISTER_MAPPER(round, ActivationMapper)
REGISTER_MAPPER(leaky_relu, LeakyReluMapper)
REGISTER_MAPPER(gelu, GeluMapper)
REGISTER_MAPPER(selu, SeluMapper)
REGISTER_MAPPER(prelu, PReluMapper)
REGISTER_MAPPER(hard_sigmoid, HardSigmoidMapper)
REGISTER_MAPPER(swish, SwishMapper)
REGISTER_MAPPER(hard_swish, HardSwishMapper)
REGISTER_MAPPER(softmax, SoftMaxMapper)

int32_t ActivationMapper::GetMinOpset(bool verbose) {
  if (OpType() == "softplus") {
    float beta = 0.0;
    float threshold = 20.0;
    GetAttr("beta", &beta);
    GetAttr("threshold", &threshold);
    if ((beta - 1.0) > 1e-06 || (beta - 1.0) < -1e-06 ||
        (threshold - 20.0) > 1e-06 || (threshold - 20.0) < -1e-06) {
      Error() << "Only support softplus with beta == 1.0 and threshold == 20.0."
              << std::endl;
      return -1;
    }
  }
  if (OpType() == "round") {
    Logger(verbose) << RequireOpset(11) << std::endl;
    return 11;
  }
  return 7;
}

void ActivationMapper::Opset7() {
  auto input_info = GetInput("X");
  auto output_info = GetOutput("Out");
  auto iter = op_mapper_.find(OpType());
  Assert(op_mapper_.end() != iter,
         "Cannot find " + OpType() + " in activation op_mapper.");
  helper_->MakeNode(iter->second, {input_info[0].name}, {output_info[0].name});
}

void Relu6Mapper::Opset7() {
  auto input_info = GetInput("X");
  auto output_info = GetOutput("Out");
  float min = 0.0;
  helper_->Clip(input_info[0].name, output_info[0].name, min, threshold_,
                input_info[0].dtype);
}

int32_t PReluMapper::GetMinOpset(bool verbose) {
  auto input_info = GetInput("X");
  auto slope_info = GetInput("Alpha");
  if (input_info[0].Rank() != slope_info[0].Rank()) {
    if (slope_info[0].Rank() > 1) {
      Error()
          << "Only support rank of alpha <=1 while Rank(alpha) != Rank(input)."
          << std::endl;
      return -1;
    }
  }
  return 7;
}

void PReluMapper::Opset7() {
  auto input_info = GetInput("X");
  auto slope_info = GetInput("Alpha");
  auto output_info = GetOutput("Out");

  std::string slope_cast_name = slope_info[0].name;
  if (slope_info[0].dtype == P2ODataType::FP64) {
    slope_cast_name = helper_->AutoCast({slope_info[0].name}, P2ODataType::FP64,
                                        P2ODataType::FP32);
  }

  if (slope_info[0].Rank() != input_info[0].Rank()) {
    Assert(slope_info[0].Rank() <= 1,
           "Paddle2ONNX: Only support rank of alpha <= 1 while rank of alpha "
           "is not equal with rank of input for operator prelu.");
    Assert(
        input_info[0].Rank() > 1,
        "Paddle2ONNX: Rank of input should greater than 2 for operator prelu.");
    std::vector<int64_t> shape_value(input_info[0].Rank() - 1, 1);
    shape_value[0] = -1;
    slope_cast_name = helper_->Reshape(slope_cast_name, shape_value);
  }

  if (input_info[0].dtype == P2ODataType::FP64) {
    std::string x_cast_name = helper_->AutoCast(
        {input_info[0].name}, P2ODataType::FP64, P2ODataType::FP32);
    auto node = helper_->MakeNode("PRelu", {x_cast_name, slope_cast_name});
    helper_->AutoCast(node->output(0), {output_info[0].name}, P2ODataType::FP32,
                      P2ODataType::FP64);
  } else {
    helper_->MakeNode("PRelu", {input_info[0].name, slope_cast_name},
                      {output_info[0].name});
  }
}

void SeluMapper::Opset7() {
  auto input_info = GetInput("X");
  auto output_info = GetOutput("Out");
  auto node =
      helper_->MakeNode("Selu", {input_info[0].name}, {output_info[0].name});
  AddAttribute(node, "alpha", alpha_);
  AddAttribute(node, "gamma", scale_);
}

void HardSigmoidMapper::Opset7() {
  auto input_info = GetInput("X");
  auto output_info = GetOutput("Out");
  auto node = helper_->MakeNode("HardSigmoid", {input_info[0].name},
                                {output_info[0].name});
  AddAttribute(node, "alpha", alpha_);
  AddAttribute(node, "beta", beta_);
}

void SwishMapper::Opset7() {
  auto input_info = GetInput("X");
  auto output_info = GetOutput("Out");

  std::string beta_node =
      helper_->Constant({1}, GetOnnxDtype(input_info[0].dtype), beta_);
  // TODO(jiangjiajun) eliminate multiply with a constant of value 1
  // TODO(jiangjiajun) eliminate add with a constant of value 0
  auto beta_x_node = helper_->MakeNode("Mul", {input_info[0].name, beta_node});
  auto sigmod_node = helper_->MakeNode("Sigmoid", {beta_x_node->output(0)});
  helper_->MakeNode("Mul", {input_info[0].name, sigmod_node->output(0)},
                    {output_info[0].name});
}

void HardSwishMapper::Opset7() {
  auto input_info = GetInput("X");
  auto output_info = GetOutput("Out");

  std::string scale_node =
      helper_->Constant({1}, GetOnnxDtype(input_info[0].dtype), scale_);
  std::string offset_node =
      helper_->Constant({1}, GetOnnxDtype(input_info[0].dtype), offset_);

  auto add_node = helper_->MakeNode("Add", {input_info[0].name, offset_node});
  auto clip_node =
      helper_->Clip(add_node->output(0), 0.0, threshold_, input_info[0].dtype);

  auto mul_node = helper_->MakeNode("Mul", {input_info[0].name, clip_node});
  helper_->MakeNode("Div", {mul_node->output(0), scale_node},
                    {output_info[0].name});
}

void LeakyReluMapper::Opset7() {
  auto input_info = GetInput("X");
  auto output_info = GetOutput("Out");
  auto node = helper_->MakeNode("LeakyRelu", {input_info[0].name},
                                {output_info[0].name});
  AddAttribute(node, "alpha", alpha_);
}

void GeluMapper::Opset9() {
  auto input_info = GetInput("X");
  auto output_info = GetOutput("Out");
  auto input_onnx_dtype = GetOnnxDtype(input_info[0].dtype);
  double sqrt_2_value = 1.4142135623730951;
  double scale_value = 0.5;
  double const_1_value = 1.0;
  auto sqrt_2 =
      helper_->Constant({1}, ONNX_NAMESPACE::TensorProto::FLOAT, sqrt_2_value);
  auto scale =
      helper_->Constant({1}, ONNX_NAMESPACE::TensorProto::FLOAT, scale_value);
  auto const_1 =
      helper_->Constant({1}, ONNX_NAMESPACE::TensorProto::FLOAT, const_1_value);

  auto input_name = helper_->AutoCast(input_info[0].name, input_info[0].dtype,
                                      P2ODataType::FP32);

  // the computation formula follows
  // https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/functional/gelu_cn.html#gelu
  auto erf0 = helper_->MakeNode("Div", {input_name, sqrt_2});
  auto erf1 = helper_->MakeNode("Erf", {erf0->output(0)});
  auto gelu0 = helper_->MakeNode("Add", {erf1->output(0), const_1});
  auto gelu1 = helper_->MakeNode("Mul", {input_name, gelu0->output(0)});

  if (input_info[0].dtype != P2ODataType::FP32) {
    auto out = helper_->MakeNode("Mul", {gelu1->output(0), scale});
    auto cast_out =
        helper_->MakeNode("Cast", {out->output(0)}, {output_info[0].name});
    AddAttribute(cast_out, "to", GetOnnxDtype(input_info[0].dtype));
  } else {
    helper_->MakeNode("Mul", {gelu1->output(0), scale}, {output_info[0].name});
  }
}

void SoftMaxMapper::Opset7() {
  auto input_info = GetInput("X");
  auto output_info = GetOutput("Out");
  if (axis_ < 0) {
    axis_ = axis_ + output_info[0].Rank();
  }
  if (axis_ == output_info[0].Rank() - 1) {
    auto node = helper_->MakeNode("Softmax", {input_info[0].name},
                                  {output_info[0].name});
    AddAttribute(node, "axis", axis_);
  } else {
    std::vector<int64_t> perm = Arange(0, output_info[0].Rank());
    perm[output_info[0].Rank() - 1] = axis_;
    perm[axis_] = output_info[0].Rank() - 1;
    auto transpose_node = helper_->MakeNode("Transpose", {input_info[0].name});
    AddAttribute(transpose_node, "perm", perm);
    auto softmax_node =
        helper_->MakeNode("Softmax", {transpose_node->output(0)});
    int64_t axis_last = -1;
    AddAttribute(softmax_node, "axis", axis_last);
    auto transpose_node_last = helper_->MakeNode(
        "Transpose", {softmax_node->output(0)}, {output_info[0].name});
    AddAttribute(transpose_node_last, "perm", perm);
  }
}

void SoftMaxMapper::Opset13() {
  int64_t axis;
  GetAttr("axis", &axis);
  auto input_info = GetInput("X");
  auto output_info = GetOutput("Out");
  auto node =
      helper_->MakeNode("Softmax", {input_info[0].name}, {output_info[0].name});
  AddAttribute(node, "axis", axis);
}

}  // namespace paddle2onnx
