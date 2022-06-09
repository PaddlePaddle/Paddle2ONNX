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

#include "paddle2onnx/mapper/tensor/eye.h"

namespace paddle2onnx {
REGISTER_MAPPER(eye, EyeMapper)

int32_t EyeMapper::GetMinOpset(bool verbose) {
  Logger(verbose, 9) << RequireOpset(9) << std::endl;
  return 9;
}

void EyeMapper::Opset9() {
  auto output_info = parser_->GetOpOutput(block_idx_, op_idx_, "Out");

  std::string constant_node = helper_->Constant(
      {num_rows_, num_columns_}, GetOnnxDtype(output_info[0].dtype), 0);

  auto node =
      helper_->MakeNode("EyeLike", {constant_node}, {output_info[0].name});
}

}  // namespace paddle2onnx
