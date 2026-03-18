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

#include "paddle2onnx/mapper/tensor/put_along_axis.h"
#include <iostream>
#include <string>
#include <vector>

namespace paddle2onnx {
REGISTER_PIR_MAPPER(put_along_axis, PutAlongAxisMapper)

int32_t PutAlongAxisMapper::GetMinOpsetVersion(bool verbose) {
  if (reduce_ != "assign" && !include_self_) {
    Error() << "Only support artt 'include_self' == 'false' when attr 'reduce' "
               "== 'assign'"
            << std::endl;
    return -1;
  }
  if (reduce_ == "mean") {
    Error() << "Not support artt 'reduce' == 'mean' yet." << std::endl;
    return -1;
  }
  if (reduce_ == "amin" || reduce_ == "amax") {
    Logger(verbose, 18) << RequireOpset(18) << std::endl;
    return 18;
  }
  if (reduce_ != "assign") {
    Logger(verbose, 16) << RequireOpset(16) << std::endl;
    return 16;
  }
  Logger(verbose, 11) << RequireOpset(11) << std::endl;
  return 11;
}

void PutAlongAxisMapper::Opset11() {
  auto arr_info = GetInput("arr");
  auto indices_info = GetInput("indices");
  auto values_info = GetInput("values");
  auto out_info = GetOutput("out");
  auto node = helper_->MakeNode(
      "ScatterElements",
      {arr_info[0].name, indices_info[0].name, values_info[0].name},
      {out_info[0].name});
  AddAttribute(node, "axis", axis_);
}

void PutAlongAxisMapper::Opset16() {
  auto arr_info = GetInput("arr");
  auto indices_info = GetInput("indices");
  auto values_info = GetInput("values");
  auto out_info = GetOutput("out");
  auto node = helper_->MakeNode(
      "ScatterElements",
      {arr_info[0].name, indices_info[0].name, values_info[0].name},
      {out_info[0].name});
  std::string onnx_reduction = "none";
  if (reduce_ == "assign") {
    onnx_reduction = "none";
  } else if (reduce_ == "add") {
    onnx_reduction = "add";
  } else if (reduce_ == "multiply") {
    onnx_reduction = "mul";
  }
  AddAttribute(node, "axis", axis_);
  AddAttribute(node, "reduction", onnx_reduction);
}
void PutAlongAxisMapper::Opset18() {
  auto arr_info = GetInput("arr");
  auto indices_info = GetInput("indices");
  auto values_info = GetInput("values");
  auto out_info = GetOutput("out");
  auto node = helper_->MakeNode(
      "ScatterElements",
      {arr_info[0].name, indices_info[0].name, values_info[0].name},
      {out_info[0].name});
  std::string onnx_reduction = "none";
  if (reduce_ == "assign") {
    onnx_reduction = "none";
  } else if (reduce_ == "add") {
    onnx_reduction = "add";
  } else if (reduce_ == "multiply") {
    onnx_reduction = "mul";
  } else if (reduce_ == "amin") {
    onnx_reduction = "min";
  } else if (reduce_ == "amax") {
    onnx_reduction = "max";
  }
  AddAttribute(node, "axis", axis_);
  AddAttribute(node, "reduction", onnx_reduction);
}
}  // namespace paddle2onnx
