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

#include "paddle2onnx/mapper/tensor/pir_select_input.h"

#include <cmath>
#include <string>
#include <vector>

namespace paddle2onnx {
REGISTER_PIR_MAPPER(select_input, PirSelectInputMapper);

int32_t PirSelectInputMapper::GetMinOpsetVersion(bool verbose) { return 9; }

void PirSelectInputMapper::Opset9() {
  auto cond_info = GetInput(0);
  auto false_info = GetInput(1);
  auto true_info = GetInput(2);
  auto out_info = GetOutput(0);

  std::string cast_cond_info = helper_->AutoCast(
      cond_info[0].name, P2ODataType::INT32, P2ODataType::BOOL);
  helper_->MakeNode("Where",
                    {cast_cond_info, true_info[0].name, false_info[0].name},
                    {out_info[0].name});
}

}  // namespace paddle2onnx
