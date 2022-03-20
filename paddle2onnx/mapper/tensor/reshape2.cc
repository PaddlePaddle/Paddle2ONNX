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

#include "paddle2onnx/mapper/tensor/reshape2.h"
#include <iostream>
#include <string>
#include <vector>

namespace paddle2onnx {
REGISTER_MAPPER(reshape2, Reshape2Mapper)

void Reshape2Mapper::Opset7(OnnxHelper* helper) {
  auto op = parser_->GetOpDesc(block_idx_, op_idx_);
  std::vector<TensorInfo> input_info =
      parser_->GetOpInput(block_idx_, op_idx_, "X");
  std::vector<TensorInfo> output_info =
      parser_->GetOpOutput(block_idx_, op_idx_, "Out");

  std::string shape_name = "ShapeTensor";
  if (!parser_->OpHasInput(block_idx_, op_idx_, shape_name)) {
    shape_name = "Shape";
  }

  std::string new_shape = "";
  if (parser_->OpHasInput(block_idx_, op_idx_, shape_name)) {
    std::vector<TensorInfo> shape_info =
        parser_->GetOpInput(block_idx_, op_idx_, shape_name);
    if (shape_info.size() > 1) {
      new_shape = helper->ConcatIndices(shape_info);
    } else {
      new_shape = helper->AutoCast(shape_info[0].name, shape_info[0].dtype,
                                   P2ODataType::INT64);
    }
  } else {
    std::vector<int64_t> value;
    parser_->GetOpAttr(op, "shape", &value);
    new_shape = helper->Constant(ONNX_NAMESPACE::TensorProto::INT64, value);
  }
  helper->MakeNode("Reshape", {input_info[0].name, new_shape},
                   {output_info[0].name});
}

}  // namespace paddle2onnx
