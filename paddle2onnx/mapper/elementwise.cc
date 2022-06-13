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
#include "paddle2onnx/mapper/elementwise.h"

namespace paddle2onnx {

REGISTER_MAPPER(elementwise_add, ElementwiseMapper)
REGISTER_MAPPER(elementwise_sub, ElementwiseMapper)
REGISTER_MAPPER(elementwise_div, ElementwiseMapper)
REGISTER_MAPPER(elementwise_mul, ElementwiseMapper)
REGISTER_MAPPER(elementwise_min, ElementwiseMapper)
REGISTER_MAPPER(elementwise_max, ElementwiseMapper)
REGISTER_MAPPER(elementwise_pow, ElementwiseMapper)
REGISTER_MAPPER(elementwise_mod, ElementWiseModMapper)
REGISTER_MAPPER(elementwise_floordiv, ElementWiseFloordivMapper)

int32_t ElementwiseMapper::GetMinOpset(bool verbose) {
  if (OpType() == "elementwise_min" || OpType() == "elementwise_max") {
    Logger(verbose, 8) << RequireOpset(8) << std::endl;
    return 8;
  }
  return 7;
}

void ElementwiseMapper::Opset7() {
  auto input_x_info = GetInput("X");
  auto input_y_info = GetInput("Y");
  auto output_info = GetOutput("Out");
  auto iter = op_mapper_.find(OpType());
  Assert(op_mapper_.end() != iter,
         "Cannot find " + OpType() + " in elementwise op_mapper.");

  if (axis_ == -1 || axis_ == (input_x_info[0].Rank() - 1) ||
      input_x_info[0].Rank() == input_y_info[0].Rank()) {
    helper_->MakeNode(iter->second,
                      {input_x_info[0].name, input_y_info[0].name},
                      {output_info[0].name});
  } else {
    std::vector<int64_t> broadcast_shape(input_x_info[0].Rank(), 1);
    for (int i = axis_; i < axis_ + input_y_info[0].Rank(); ++i) {
      broadcast_shape[i] = input_y_info[0].shape[i - axis_];
    }
    std::string broadcast_shape_node =
        helper_->Constant(GetOnnxDtype(P2ODataType::INT64), broadcast_shape);
    auto y_node = helper_->MakeNode(
        "Reshape", {input_y_info[0].name, broadcast_shape_node});
    helper_->MakeNode(iter->second, {input_x_info[0].name, y_node->output(0)},
                      {output_info[0].name});
  }
}

void ElementWiseModMapper::Opset10() {
  auto input_x_info = GetInput("X");
  auto input_y_info = GetInput("Y");
  auto output_info = GetOutput("Out");
  int64_t fmod = 0;
  if (input_y_info[0].dtype == P2ODataType::INT32 ||
      input_y_info[0].dtype == P2ODataType::INT64) {
    auto mod_node =
        helper_->MakeNode("Mod", {input_x_info[0].name, input_y_info[0].name},
                          {output_info[0].name});
    AddAttribute(mod_node, "fmod", fmod);
    return;
  }

  fmod = 1;

  auto abs_x_node = helper_->MakeNode("Abs", {input_x_info[0].name});
  auto abs_y_node = helper_->MakeNode("Abs", {input_y_info[0].name});

  auto dtype = input_y_info[0].dtype;
  std::vector<float> val_0 = {0.0};

  std::string zero_node = helper_->Constant(GetOnnxDtype(dtype), val_0);

  auto mod_node =
      helper_->MakeNode("Mod", {abs_x_node->output(0), abs_y_node->output(0)});
  AddAttribute(mod_node, "fmod", fmod);

  auto neg_node = helper_->MakeNode("Neg", {mod_node->output(0)});

  auto less_node = helper_->MakeNode("Less", {input_x_info[0].name, zero_node});

  std::string condition_node =
      helper_->AutoCast(less_node->output(0), dtype, P2ODataType::BOOL);

  auto mod_res_node = helper_->MakeNode(
      "Where", {condition_node, neg_node->output(0), mod_node->output(0)});

  auto mod_y_add_node =
      helper_->MakeNode("Add", {mod_res_node->output(0), input_y_info[0].name});

  auto mod_y_mul_node =
      helper_->MakeNode("Mul", {mod_res_node->output(0), input_y_info[0].name});

  auto mod_y_mul_less_node =
      helper_->MakeNode("Less", {mod_y_mul_node->output(0), zero_node});

  std::string mod_y_mul_condition_node = helper_->AutoCast(
      mod_y_mul_less_node->output(0), dtype, P2ODataType::BOOL);

  helper_->MakeNode("Where",
                    {mod_y_mul_condition_node, mod_y_add_node->output(0),
                     mod_res_node->output(0)},
                    {output_info[0].name});
}

void ElementWiseFloordivMapper::Opset7() {
  auto input_x_info = GetInput("X");
  auto input_y_info = GetInput("Y");
  auto output_info = GetOutput("Out");

  // // Pod Types
  // BOOL = 0;
  // INT16 = 1;
  // INT32 = 2;
  // INT64 = 3;
  // FP16 = 4;
  // FP32 = 5;
  // FP64 = 6;
  // // Tensor<size_t> is used in C++.
  // SIZE_T = 19;
  // UINT8 = 20;
  // INT8 = 21;
  // BF16 = 22;
  // COMPLEX64 = 23;
  // COMPLEX128 = 24;

  bool is_int = false;
  if (input_x_info[0].dtype <= 3 ||
      input_x_info[0].dtype == P2ODataType::UINT8 ||
      input_x_info[0].dtype == P2ODataType::INT8 ||
      input_y_info[0].dtype <= 3 ||
      input_y_info[0].dtype == P2ODataType::UINT8 ||
      input_y_info[0].dtype == P2ODataType::INT8) {
    is_int = true;
  }

  P2OLogger() << "floor div is int: " << is_int << std::endl;

  Assert(axis_ == -1, "floor div axis is fixed to -1");

  if (is_int) {
    // Integer division does trunction rounding
    auto div = helper_->MakeNode(
        "Div", {input_x_info[0].name, input_y_info[0].name}, 1);
    auto zero =
        helper_->Constant<int64_t>(GetOnnxDtype(P2ODataType::INT64), {0});

    // compute negtive
    auto opset_ver = helper_->GetOpsetVersion();
    auto self = input_x_info[0];
    auto other = input_y_info[0];
    std::string x = self.name;
    std::string y = other.name;
    std::shared_ptr<ONNX_NAMESPACE::NodeProto> self_lt_zero, other_lt_zero;
    if (opset_ver <= 8) {
      x = helper_->AutoCast(self.name, self.dtype, P2ODataType::FP32);
      y = helper_->AutoCast(self.name, self.dtype, P2ODataType::FP32);

      self_lt_zero = helper_->MakeNode("Less", {x, zero}, 1);
      other_lt_zero = helper_->MakeNode("Less", {y, zero}, 1);
    }
    {
      // opset 9
      if (self.dtype == P2ODataType::BOOL && other.dtype == P2ODataType::BOOL) {
        x = helper_->AutoCast(self.name, self.dtype, P2ODataType::INT32);
        y = helper_->AutoCast(other.name, other.dtype, P2ODataType::INT32);
      }

      self_lt_zero = helper_->MakeNode("Less", {x, zero}, 1);
      other_lt_zero = helper_->MakeNode("Less", {y, zero}, 1);
    }

    auto negtive = helper_->MakeNode(
        "Xor", {self_lt_zero->output(0), other_lt_zero->output(0)}, 1);

    auto div_other = helper_->MakeNode("Mul", {div->output(0), other.name}, 1);
    auto mod = helper_->MakeNode("Sub", {self.name, div_other->output(0)}, 1);

    auto mod_eq_zero = helper_->MakeNode("Equal", {mod->output(0), zero}, 1);
    auto not_eq_zero = helper_->MakeNode("Not", {mod_eq_zero->output(0)}, 1);
    auto fixup_mask = helper_->MakeNode(
        "And", {negtive->output(0), not_eq_zero->output(0)}, 1);

    auto fixup_mask_int64 = helper_->AutoCast(
        fixup_mask->output(0), P2ODataType::BOOL, P2ODataType::INT64);

    auto one =
        helper_->Constant<int64_t>(GetOnnxDtype(P2ODataType::INT64), {1});
    auto fixup = helper_->MakeNode("Mul", {fixup_mask_int64, one}, 1);

    helper_->MakeNode("Sub", {div->output(0), fixup->output(0)},
                      {output_info[0].name});

  } else {
    // float
    auto div_node =
        helper_->MakeNode("Div", {input_x_info[0].name, input_y_info[0].name});
    helper_->MakeNode("Floor", {div_node->output(0)}, {output_info[0].name});
  }
}

}  // namespace paddle2onnx
