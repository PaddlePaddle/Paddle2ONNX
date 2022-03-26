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

#include "paddle2onnx/mapper/tensor/tensor_array_to_tensor.h"

namespace paddle2onnx {
REGISTER_MAPPER(tensor_array_to_tensor, TensorArrayToTensorMapper)

void TensorArrayToTensorMapper::Opset7(OnnxHelper* helper) {
  auto input_info = GetInput("X");
  auto output_info = GetOutput("Out");

  if (axis_ == 0) {
    if (output_info[0].Rank() == 1) {
      helper->Reshape(input_info[0].name, output_info[0].name, {-1});
      return;
    }
    auto shape =
        helper->MakeNode("Shape", {input_info[0].name}, {output_info[0].name})
            ->output(0);
    auto part2 = helper->Slice(shape, {0}, {2}, {output_info[0].Rank() + 1});
    auto unknown_dim = helper->Constant(ONNX_NAMESPACE::TensorProto::INT64,
                                        std::vector<int64_t>(1, -1));
    auto new_shape = helper->Concat({unknown_dim, part2}, 0);
    helper->MakeNode("Reshape", {input_info[0].name, new_shape},
                     {output_info[0].name});
  } else {
    std::vector<int64_t> perm = Arange(0, output_info[0].Rank() + 1);
    auto axis = axis_;
    if (axis < 0) {
      axis += output_info[0].Rank();
    }
    perm[axis + 1] = 1;
    perm[1] = axis + 1;
    auto input_t = helper->MakeNode("Transpose", {input_info[0].name});
    AddAttribute(input_t, "perm", perm);
    auto shape = helper->MakeNode("Shape", {input_t->output(0)})->output(0);
    auto part2 = helper->Slice(shape, {0}, {2}, {output_info[0].Rank() + 1});
    auto unknown_dim = helper->Constant(ONNX_NAMESPACE::TensorProto::INT64,
                                        std::vector<int64_t>(1, -1));
    auto new_shape = helper->Concat({unknown_dim, part2}, 0);
    auto out =
        helper->MakeNode("Reshape", {input_info[0].name, new_shape})->output(0);
    std::vector<int64_t> out_perm = Arange(0, output_info[0].Rank());
    out_perm[0] = axis;
    out_perm[axis] = 0;
    auto out_t = helper->MakeNode("Transpose", {out}, {output_info[0].name});
    AddAttribute(out_t, "perm", perm);
  }
}

}  // namespace paddle2onnx
