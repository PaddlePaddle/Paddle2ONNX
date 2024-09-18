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
#include "paddle2onnx/mapper/tensor/elementwise_add.h"

namespace paddle2onnx {
REGISTER_MAPPER(elementwise_add, ElementwiseAddMapper)

int32_t ElementwiseAddMapper::GetMinOpsetVersion(bool verbose) {
  auto input_x_info = GetInput("X");
  auto input_y_info = GetInput("Y");

  auto x_name = input_x_info[0].name;
  auto y_name = input_y_info[0].name;

  if ((input_x_info[0].dtype == P2ODataType::UINT8) || (input_y_info[0].dtype == P2ODataType::UINT8)) {
    Logger(verbose, 14) << RequireOpset(14) << std::endl;
    return 14;
  }

  if ((input_x_info[0].dtype == P2ODataType::INT8) || (input_y_info[0].dtype == P2ODataType::INT8)) {
    Logger(verbose, 14) << RequireOpset(14) << std::endl;
    return 14;
  }
  return 7;
}

void ElementwiseAddMapper::ExportForONNX() {
  auto input_x_info = GetInput("X");
  auto input_y_info = GetInput("Y");
  auto output_info = GetOutput("Out");

  auto x_name = input_x_info[0].name;
  auto y_name = input_y_info[0].name;
  if (input_x_info[0].dtype == P2ODataType::BOOL && input_y_info[0].dtype == P2ODataType::BOOL) {
    x_name = helper_->AutoCast(x_name, input_x_info[0].dtype, P2ODataType::INT32);
    y_name = helper_->AutoCast(y_name, input_y_info[0].dtype, P2ODataType::INT32);
  }

  std::string output_name;
  if (axis_ == -1 || axis_ == (input_x_info[0].Rank() - 1) || input_x_info[0].Rank() == input_y_info[0].Rank()) {
    output_name = helper_->MakeNode("Add", {x_name, y_name})->output(0);
  } else {
    std::vector<int64_t> broadcast_shape(input_x_info[0].Rank(), 1);
    for (int i = axis_; i < axis_ + input_y_info[0].Rank(); ++i) {
      broadcast_shape[i] = input_y_info[0].shape[i - axis_];
    }
    std::string broadcast_shape_node = helper_->Constant(GetOnnxDtype(P2ODataType::INT64), broadcast_shape);
    auto y_node = helper_->MakeNode("Reshape", {y_name, broadcast_shape_node});
    output_name = helper_->MakeNode("Add", {x_name, y_node->output(0)})->output(0);
  }

  if (input_x_info[0].dtype == P2ODataType::BOOL && input_y_info[0].dtype == P2ODataType::BOOL) {
    helper_->AutoCast(output_name, output_info[0].name, P2ODataType::INT32, P2ODataType::BOOL);
  } else {
    helper_->MakeNode("Identity", {output_name}, {output_info[0].name});
  }
}

void ElementwiseAddMapper::ExportForRKNN() {
  auto input_x_info = GetInput("X");
  auto input_y_info = GetInput("Y");
  auto output_info = GetOutput("Out");

  auto x_name = input_x_info[0].name;
  auto y_name = input_y_info[0].name;
  if (input_x_info[0].dtype == P2ODataType::BOOL && input_y_info[0].dtype == P2ODataType::BOOL) {
    x_name = helper_->AutoCast(x_name, input_x_info[0].dtype, P2ODataType::INT32);
    y_name = helper_->AutoCast(y_name, input_y_info[0].dtype, P2ODataType::INT32);
  }

  std::string output_name;
  do {
    if ((axis_ == -1) && (input_y_info[0].dtype == P2ODataType::FP32) && (input_y_info[0].shape.size() == 1))
    {
      axis_ = input_x_info[0].shape.size() - 1;
      std::vector<int64_t> broadcast_shape(input_x_info[0].Rank(), 1);
      for (int i = axis_; i < axis_ + input_y_info[0].Rank(); ++i) {
        broadcast_shape[i] = input_y_info[0].shape[i - axis_];
      }
      std::vector<float> values;
      if (TryGetValue(input_y_info[0], &values))
      {
        y_name = helper_->Constant(broadcast_shape, GetOnnxDtype(P2ODataType::FP32), values);
        output_name = helper_->MakeNode("Add", {x_name, y_name})->output(0);
        break;
      }
    }
  
    if (axis_ == -1 || axis_ == (input_x_info[0].Rank() - 1) || input_x_info[0].Rank() == input_y_info[0].Rank()) {
      output_name = helper_->MakeNode("Add", {x_name, y_name})->output(0);
    } else {
      std::vector<int64_t> broadcast_shape(input_x_info[0].Rank(), 1);
      for (int i = axis_; i < axis_ + input_y_info[0].Rank(); ++i) {
        broadcast_shape[i] = input_y_info[0].shape[i - axis_];
      }
      std::string broadcast_shape_node = helper_->Constant(GetOnnxDtype(P2ODataType::INT64), broadcast_shape);
      auto y_node = helper_->MakeNode("Reshape", {y_name, broadcast_shape_node});
      output_name = helper_->MakeNode("Add", {x_name, y_node->output(0)})->output(0);
    }
  } while (0);

  if (input_x_info[0].dtype == P2ODataType::BOOL && input_y_info[0].dtype == P2ODataType::BOOL) {
    helper_->AutoCast(output_name, output_info[0].name, P2ODataType::INT32, P2ODataType::BOOL);
  } else {
    helper_->MakeNode("Identity", {output_name}, {output_info[0].name});
  }
}

void ElementwiseAddMapper::Opset7() {
  if (this->deploy_backend == "rknn") {
    return ExportForRKNN();
  } else {
    return ExportForONNX();
  }
}
}