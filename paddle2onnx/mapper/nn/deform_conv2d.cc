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

#include "paddle2onnx/mapper/nn/deform_conv2d.h"

#include <string>
#include <vector>

namespace paddle2onnx
{
  REGISTER_MAPPER(deformable_conv, DeformConv2dMapper)

  int32_t DeformConv2dMapper::GetMinOpset(bool verbose)
  {
    return 7;
  }

  void DeformConv2dMapper::Opset7()
  {
    auto kernel_info = GetInput("Filter");
    auto input_info = GetInput("Input");
    auto offset_info = GetInput("Offset");
    auto mask_info = GetInput("Mask");
    auto output_info = GetOutput("Output");

    // TODO
    // 1. 这个变量名称映射paddle的是哪个呢？看上去是ops.yaml和op_compat.yaml吗？
    // 2. c++ api上没有bias，这个是def deformable_conv调用C_ops后再加上的，这里怎么处理呢？optional的输入也要全写入吗
    // 3. 这个opsetx怎么具体对应起来呢？跟onnx似乎对不上
    auto node = helper_->MakeNode(
        "DeformConv", {input_info[0].name, kernel_info[0].name, offset_info[0].name}, {output_info[0].name});

    AddAttribute(node, "dilations", dilations_);
    AddAttribute(node, "group", groups_);
    std::vector<int64_t> kernel_shape = {kernel_info[0].shape[2],
                                         kernel_info[0].shape[3]};
    AddAttribute(node, "kernel_shape", kernel_shape);
    AddAttribute(node, "pads", paddings_);
    AddAttribute(node, "strides", strides_);
  }

} // namespace paddle2onnx
