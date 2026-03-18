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

#include "paddle2onnx/mapper/tensor/tensor_array.h"
#include <iostream>
#include <string>
#include <vector>
#include "paddle2onnx/mapper/onnx_helper.h"

namespace paddle2onnx {
REGISTER_PIR_MAPPER(create_array, CreateArrayMapper)
REGISTER_PIR_MAPPER(array_length, ArrayLengthMapper)
REGISTER_PIR_MAPPER(array_write_, ArrayWriteMapper)
REGISTER_PIR_MAPPER(array_read, ArrayReadMapper)

int32_t CreateArrayMapper::GetMinOpsetVersion(bool verbose) {
  Logger(verbose, 11) << "CreateArrayMapper " << RequireOpset(11) << std::endl;
  return 11;
}

void CreateArrayMapper::Opset11() {
  auto output_info = GetOutput(0);
  auto node = helper_->MakeNode("SequenceEmpty", {}, {output_info[0].name});
  AddAttribute(node, "dtype", GetOnnxDtype(dtype_));
  SetTensorArrayName(output_info[0].name);
}
int32_t ArrayLengthMapper::GetMinOpsetVersion(bool verbose) {
  Logger(verbose, 11) << "ArrayLengthMapper " << RequireOpset(11) << std::endl;
  return 11;
}
void ArrayLengthMapper::Opset11() {
  auto output_info = GetOutput(0);
  std::string arr_name = GetTensorArrayName();
  helper_->MakeNode("SequenceLength", {arr_name}, {output_info[0].name});
}
int32_t ArrayWriteMapper::GetMinOpsetVersion(bool verbose) {
  Logger(verbose, 11) << "ArrayWriteMapper " << RequireOpset(11) << std::endl;
  return 11;
}
void ArrayWriteMapper::Opset11() {
  auto tensor_info = GetInput(1);
  auto index_info = GetInput(2);
  auto output_info = GetOutput(0);
  auto squeeze_node = helper_->MakeNode("Squeeze", {index_info[0].name});
  std::string arr_name = GetTensorArrayName();
  helper_->MakeNode("SequenceInsert",
                    {arr_name, tensor_info[0].name, squeeze_node->output(0)},
                    {output_info[0].name});
  SetTensorArrayName(output_info[0].name);
}
int32_t ArrayReadMapper::GetMinOpsetVersion(bool verbose) {
  Logger(verbose, 11) << "ArrayReadMapper " << RequireOpset(11) << std::endl;
  return 11;
}
void ArrayReadMapper::Opset11() {
  auto index_info = GetInput(1);
  auto output_info = GetOutput(0);
  auto squeeze_node = helper_->MakeNode("Squeeze", {index_info[0].name});
  std::string arr_name = GetTensorArrayName();
  helper_->MakeNode(
      "SequenceAt", {arr_name, squeeze_node->output(0)}, {output_info[0].name});
}

}  // namespace paddle2onnx
