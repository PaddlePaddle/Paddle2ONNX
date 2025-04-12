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

#include "paddle2onnx/mapper/tensor/abs.h"

namespace paddle2onnx {
REGISTER_PIR_MAPPER(abs, AbsMapper)
REGISTER_MAPPER(abs, AbsMapper)

int32_t AbsMapper::GetMinOpsetVersion(bool verbose) { return 13; }

void AbsMapper::Opset13() {
  auto input_info = GetInput("X");
  auto output_info = GetOutput("Out");
  if (input_info[0].dtype == P2ODataType::COMPLEX64) {
    std::string one_str = helper_->Constant(GetOnnxDtype(P2ODataType::INT64),
                                            std::vector<int64_t>({1}));
    auto split_node = helper_->MakeNode("Split", {input_info[0].name}, 2);
    AddAttribute(split_node, "axis", int64_t(-1));
    std::string split_node1 = helper_->Squeeze(split_node->output(0), {-1});
    std::string split_node2 = helper_->Squeeze(split_node->output(1), {-1});
    auto real_squre = helper_->MakeNode("Mul", {split_node1, split_node1});
    auto imag_squre = helper_->MakeNode("Mul", {split_node2, split_node2});
    auto node_add = helper_->MakeNode(
        "Add", {real_squre->output(0), imag_squre->output(0)});
    helper_->MakeNode("Sqrt", {node_add->output(0)}, {output_info[0].name});
  } else {
    helper_->MakeNode("Abs", {input_info[0].name}, {output_info[0].name});
  }
}
void AbsMapper::Opset18() {
  auto input_info = GetInput("X");
  auto output_info = GetOutput("Out");
  if (input_info[0].dtype == P2ODataType::COMPLEX64) {
    std::string one_str = helper_->Constant(GetOnnxDtype(P2ODataType::INT64),
                                            std::vector<int64_t>({1}));
    auto split_node = helper_->MakeNode("Split", {input_info[0].name}, 2);
    AddAttribute(split_node, "axis", int64_t(-1));
    AddAttribute(split_node, "num_outputs", int64_t(2));
    std::string split_node1 = helper_->Squeeze(split_node->output(0), {-1});
    std::string split_node2 = helper_->Squeeze(split_node->output(1), {-1});
    auto real_squre = helper_->MakeNode("Mul", {split_node1, split_node1});
    auto imag_squre = helper_->MakeNode("Mul", {split_node2, split_node2});
    auto node_add = helper_->MakeNode(
        "Add", {real_squre->output(0), imag_squre->output(0)});
    helper_->MakeNode("Sqrt", {node_add->output(0)}, {output_info[0].name});
  } else {
    helper_->MakeNode("Abs", {input_info[0].name}, {output_info[0].name});
  }
}

}  // namespace paddle2onnx
