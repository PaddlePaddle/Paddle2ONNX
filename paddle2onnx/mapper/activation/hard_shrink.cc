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

#include "paddle2onnx/mapper/activation/hard_shrink.h"

namespace paddle2onnx {
REGISTER_MAPPER(hard_shrink, HardShrinkMapper)

int32_t HardShrinkMapper::GetMinOpsetVersion(bool verbose) {
    Logger(verbose, 9) << RequireOpset(9) << std::endl;
    return 9;
}

void HardShrinkMapper::Opset9() {
  auto node = helper_->MakeNode("Shrink", {GetInput("X")[0].name},
                                {GetOutput("Out")[0].name});
  AddAttribute(node, "lambd", threshold_);
  AddAttribute(node, "bias", float(0.0));
}
}