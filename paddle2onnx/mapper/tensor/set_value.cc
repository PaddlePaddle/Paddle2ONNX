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

#include "paddle2onnx/mapper/tensor/set_value.h"

namespace paddle2onnx {
REGISTER_MAPPER(set_value, SetValueMapper)

int32_t SetValueMapper::GetMinOpset(bool verbose) {
  if (none_axes_.size() > 0) {
    if (verbose) {
      std::cerr << "[Paddle2ONNX] Attribute none_axes is not support in op "
                   "set_value yet."
                << std::endl;
    }
    return -1;
  }
  if (axes_.size() > 1 || axes_[0] != 0) {
    if (verbose) {
      std::cerr << "[Paddle2ONNX] Attribute axes is supported while it "
                   "contains 0 only in op set_value."
                << std::endl;
    }
    return -1;
  }
  if (steps_.size() > 1 || steps_[0] != 1) {
    if (verbose) {
      std::cerr << "[Paddle2ONNX] Attribute steps is supported while it "
                   "contains 1 only in op set_value."
                << std::endl;
    }
    return -1;
  }
  if (HasInput("StepsTensorList")) {
    if (verbose) {
      std::cerr << "[Paddle2ONNX] Input StepsTensorList is not supported in op "
                   "set_value yet."
                << std::endl;
    }
    return -1;
  }
  if (GetInput("Input")[0].dtype == P2ODataType::BOOL) {
    if (verbose) {
      std::cerr << "[Paddle2ONNX] Input X with data type of boolean is not "
                   "supported in op set_value."
                << std::endl;
    }
    return -1;
  }
  return 11;
}

void SetValueMapper::Opset11(OnnxHelper* helper) {
  auto input_info = GetInput("Input");
  auto output_info = GetOutput("Out");
  std::string starts = "";
  if (HasInput("StartsTensorList")) {
    // if negtive value exists, not supported
    starts = helper->ConcatIndices(GetInput("StartsTensorList"));
  } else {
    starts = helper->Constant(ONNX_NAMESPACE::TensorProto::INT64, starts_);
  }
  std::string ends = "";
  if (HasInput("EndsTensorList")) {
    ends = helper->ConcatIndices(GetInput("EndsTensorList"));
  } else {
    // if out of range value in end exists, not supported
    ends = helper->Constant(ONNX_NAMESPACE::TensorProto::INT64, ends_);
  }
  std::cout << "00000000" << std::endl;
  std::string value = "";
  if (HasInput("ValueTensor")) {
    value = GetInput("ValueTensor")[0].name;
  } else {
    int in_dtype = input_info[0].dtype;
    if (in_dtype == P2ODataType::INT32 || in_dtype == P2ODataType::INT64) {
      value = helper->Assign(GetOnnxDtype(output_info[0].dtype), shape_,
                             int_values_);
    } else if (in_dtype == P2ODataType::FP32) {
      value = helper->Assign(GetOnnxDtype(output_info[0].dtype), shape_,
                             fp32_values_);
    } else if (in_dtype == P2ODataType::FP64) {
      value = helper->Assign(GetOnnxDtype(output_info[0].dtype), shape_,
                             fp64_values_);
    }
  }

  std::cout << "00000001" << std::endl;
  std::string steps = helper->Constant(ONNX_NAMESPACE::TensorProto::INT64,
                                       std::vector<int64_t>(axes_.size(), 1));
  std::string axes =
      helper->Constant(ONNX_NAMESPACE::TensorProto::INT64, axes_);
  auto sliced_data =
      helper->MakeNode("Slice", {input_info[0].name, starts, ends, axes, steps})
          ->output(0);
  auto sliced_shape = helper->MakeNode("Shape", {sliced_data})->output(0);
  auto expand_value =
      helper->MakeNode("Expand", {value, sliced_shape})->output(0);

  std::cout << "00000002" << std::endl;
  auto one =
      helper->Constant({}, ONNX_NAMESPACE::TensorProto::INT64, int64_t(1));
  std::cout << "???????" << one << std::endl;
  auto indices = helper
                     ->MakeNode("Range", {helper->Squeeze(starts, {}),
                                          helper->Squeeze(ends, {}), one})
                     ->output(0);
  indices = helper->Unsqueeze(indices, {1});
  std::cout << "00000003" << std::endl;

  helper->MakeNode("ScatterND", {input_info[0].name, indices, expand_value},
                   {output_info[0].name});
}

}  // namespace paddle2onnx
