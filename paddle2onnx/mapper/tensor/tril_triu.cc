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

#include "paddle2onnx/mapper/tensor/tril_triu.h"

namespace paddle2onnx {
REGISTER_MAPPER(tril_triu, TrilTriuMapper)
REGISTER_PIR_MAPPER(tril, TrilMapper)
REGISTER_PIR_MAPPER(triu, TriuMapper)

int32_t TrilTriuMapper::GetMinOpsetVersion(bool verbose) {
  constexpr int op_version = 14;
  Logger(verbose, op_version) << RequireOpset(op_version) << std::endl;
  return op_version;
}

void TrilTriuMapper::Opset14() {
  auto x_info = GetInput("X");
  auto out_info = GetOutput("Out");

  std::vector<int64_t> diagonal_vec{diagonal_};

  std::string diagonal_node_name =
      helper_->Constant(ONNX_NAMESPACE::TensorProto::INT64, diagonal_vec);
  auto output_node = helper_->MakeNode(
      "Trilu", {x_info[0].name, diagonal_node_name}, {out_info[0].name});
  int64_t upper = !lower_;
  AddAttribute(output_node, "upper", upper);
}

int32_t TrilMapper::GetMinOpsetVersion(bool verbose) {
  constexpr int op_version = 14;
  Logger(verbose, op_version) << RequireOpset(op_version) << std::endl;
  return op_version;
}

void TrilMapper::Opset14() {
  auto x_info = GetInput("x");
  auto out_info = GetOutput("out");

  std::vector<int64_t> diagonal_vec{diagonal_};

  std::string diagonal_node_name =
      helper_->Constant(ONNX_NAMESPACE::TensorProto::INT64, diagonal_vec);
  auto output_node = helper_->MakeNode(
      "Trilu", {x_info[0].name, diagonal_node_name}, {out_info[0].name});
  int64_t upper = !lower_;
  AddAttribute(output_node, "upper", upper);
}

int32_t TriuMapper::GetMinOpsetVersion(bool verbose) {
  constexpr int op_version = 14;
  Logger(verbose, op_version) << RequireOpset(op_version) << std::endl;
  return op_version;
}

void TriuMapper::Opset14() {
  auto x_info = GetInput("x");
  auto out_info = GetOutput("out");

  std::vector<int64_t> diagonal_vec{diagonal_};

  std::string diagonal_node_name =
      helper_->Constant(ONNX_NAMESPACE::TensorProto::INT64, diagonal_vec);
  auto output_node = helper_->MakeNode(
      "Trilu", {x_info[0].name, diagonal_node_name}, {out_info[0].name});
  int64_t upper = !lower_;
  AddAttribute(output_node, "upper", upper);
}

}  // namespace paddle2onnx
