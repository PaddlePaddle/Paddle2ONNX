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

#include "paddle2onnx/mapper/tensor/full_with_tensor.h"
#include <iostream>
#include <string>
#include <vector>
#include "paddle2onnx/proto/p2o_paddle.pb.h"

namespace paddle2onnx {
REGISTER_PIR_MAPPER(full_with_tensor, FullWithTensorMapper)

int32_t FullWithTensorMapper::GetMinOpsetVersion(bool verbose) { return 8; }

void FullWithTensorMapper::Opset8() {
  auto value_info = GetInput("value");
  auto shape_info = GetInput("shape");
  auto output_info = GetOutput("out");

  auto shape = helper_->AutoCast(
      shape_info[0].name, shape_info[0].dtype, P2ODataType::INT64);

  if (shape_info[0].Rank() == 0) {
    shape = helper_->Reshape(shape, {1});
  }

  auto expand_node = helper_->MakeNode("Expand", {value_info[0].name, shape});

  helper_->AutoCast(expand_node->output(0),
                    output_info[0].name,
                    value_info[0].dtype,
                    output_info[0].dtype);

  // double fill_value = 0;
  // if(TryGetInputValue("value", &fill_value)) {
  //    helper_->ConstOfShape(shape_info[0].name,
  //                          output_info[0].name,
  //                          GetOnnxDtype(output_info[0].dtype),
  //                          fill_value);
  // }
  // else {
  //     if(value_info[0].dtype ==
  //     paddle2onnx::framework::proto::VarType_Type_FP32) {
  //         std::vector<float> value;
  //         TryGetInputValue("value", &value);
  //         helper_->ConstOfShape(shape_info[0].name,
  //                   output_info[0].name,
  //                   GetOnnxDtype(output_info[0].dtype),
  //                   value[0]);
  //     } else if (value_info[0].dtype ==
  //     paddle2onnx::framework::proto::VarType_Type_FP64) {
  //         std::vector<double> value;
  //         TryGetInputValue("value", &value);
  //         helper_->ConstOfShape(shape_info[0].name,
  //                   output_info[0].name,
  //                   GetOnnxDtype(output_info[0].dtype),
  //                   value[0]);
  //     } else if (value_info[0].dtype ==
  //     paddle2onnx::framework::proto::VarType_Type_INT32) {
  //         std::vector<int32_t> value;
  //         TryGetInputValue("value", &value);
  //         helper_->ConstOfShape(shape_info[0].name,
  //                   output_info[0].name,
  //                   GetOnnxDtype(output_info[0].dtype),
  //                   value[0]);
  //     } else if (value_info[0].dtype ==
  //     paddle2onnx::framework::proto::VarType_Type_INT64) {
  //         std::vector<int64_t> value;
  //         TryGetInputValue("value", &value);
  //         helper_->ConstOfShape(shape_info[0].name,
  //                   output_info[0].name,
  //                   GetOnnxDtype(output_info[0].dtype),
  //                   value[0]);
  //     } else {
  //         std::cerr << "unsupported dtype for full_with_tensor, only support
  //         float32/float64/int32/int64 now." << std::endl;
  //     }
  // }
}

}  // namespace paddle2onnx
