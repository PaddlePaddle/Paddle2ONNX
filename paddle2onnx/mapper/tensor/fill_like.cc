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
#include "paddle2onnx/mapper/tensor/fill_like.h"

namespace paddle2onnx {

REGISTER_MAPPER(fill_any_like, FillLikeMapper)
REGISTER_MAPPER(fill_zeros_like, FillLikeMapper)

void FillLikeMapper::Opset9(OnnxHelper* helper) {
  std::vector<TensorInfo> input_info =
      parser_->GetOpInput(block_idx_, op_idx_, "X");
  std::vector<TensorInfo> output_info =
      parser_->GetOpOutput(block_idx_, op_idx_, "Out");
  auto op = parser_->GetOpDesc(block_idx_, op_idx_);
  auto shape_node = helper->MakeNode("Shape", {input_info[0].name});
  int64_t dtype = output_info[0].dtype;
  if (parser_->OpHasAttr(op, "dtype")) {
    parser_->GetOpAttr(op, "dtype", &dtype);
  }
  helper->ConstOfShape(shape_node->output(0), output_info[0].name,
                       GetOnnxDtype(dtype), value_);
}

}  // namespace paddle2onnx
