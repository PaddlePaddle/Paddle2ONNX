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

int32_t Reshape2Mapper::GetMinOpset(bool verbose) {
  // First set shape_name to ShapeTensor, if ShapeTensor is not in the input,
  // set it to Shape
  std::string shape_name = "ShapeTensor";
  bool is_shapetensor = parser_->OpHasInput(block_idx_, op_idx_, shape_name);
  if (!is_shapetensor) {
    shape_name = "Shape";
  }

  bool is_shape = parser_->OpHasInput(block_idx_, op_idx_, shape_name);
  if (!is_shape) {
    auto op = parser_->GetOpDesc(block_idx_, op_idx_);
    if (!parser_->OpHasAttr(op, "shape")) {
      std::cerr << "shape tensor and shape attrubite all unkown in operator: " +
                       op.type()
                << std::endl;
      return -1;
    }
  } else {
    return 7;
  }
  return 7;
}

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

  if (parser_->OpHasInput(block_idx_, op_idx_, shape_name)) {
    std::vector<TensorInfo> tensor_info =
        parser_->GetOpInput(block_idx_, op_idx_, shape_name);
    if (tensor_info.size() > 1) {
      std::vector<std::string> dims;
      for (auto i = 0; i < tensor_info.size(); i++) {
        std::string input_name = helper->AutoCast(
            tensor_info[i].name, tensor_info[i].dtype, P2ODataType::INT64);
        dims.push_back(input_name);
      }
      auto concat_node = helper->MakeNode("Concat", dims);
      int64_t axis = -1;
      AddAttribute(concat_node, "axis", axis);
      helper->MakeNode("Reshape", {input_info[0].name, concat_node->output(0)},
                       {output_info[0].name});
    }
    if (tensor_info.size() == 1) {
      std::string cast_node = helper->AutoCast(
          tensor_info[0].name, tensor_info[0].dtype, P2ODataType::INT64);
      helper->MakeNode("Reshape", {input_info[0].name, cast_node},
                       {output_info[0].name});
    }
  } else {
    std::vector<int64_t> shape;
    parser_->GetOpAttr(op, "shape", &shape);
    std::string shape_node =
        helper->Constant(GetOnnxDtype(P2ODataType::INT64), shape);
    helper->MakeNode("Reshape", {input_info[0].name, shape_node},
                     {output_info[0].name});
  }
}

}  // namespace paddle2onnx
