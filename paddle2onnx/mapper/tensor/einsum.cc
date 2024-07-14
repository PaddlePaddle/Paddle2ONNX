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

#include "paddle2onnx/mapper/tensor/einsum.h"

namespace paddle2onnx {
REGISTER_MAPPER(einsum, EinsumMapper)

int32_t EinsumMapper::GetMinOpsetVersion(bool verbose)
{
  constexpr int op_version = 12;
  Logger(verbose, op_version) << RequireOpset(op_version) << std::endl;
  return op_version;
}

void EinsumMapper::Opset12() {
  auto input_info = GetInput("Operands");
  auto output_info = GetOutput("Out");
  GetAttr("equation", &equation_);

  std::vector<std::string> input_info_names;
  for (size_t i = 0; i < input_info.size(); i++)
  {
    input_info_names.emplace_back(input_info[i].name);
  }

  std::vector<std::string> output_info_names;
  for (size_t i = 0; i < output_info.size(); i++)
  {
    output_info_names.emplace_back(output_info[i].name);
  }
  auto node = helper_->MakeNode("Einsum", input_info_names, output_info_names);
  AddAttribute(node, "equation", equation_);
}

}  // namespace paddle2onnx
