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

#include "paddle2onnx/mapper/nn/conv2d.h"

#include <string>
#include <vector>

namespace paddle2onnx {
REGISTER_MAPPER(conv2d, Conv2dMapper)
REGISTER_MAPPER(depthwise_conv2d, Conv2dMapper)

int32_t Conv2dMapper::GetMinOpset(bool verbose) {
  // NHWC is not supported
  if (data_format_ == "NHWC") {
    Error() << "Cannot support input with NHWC format." << std::endl;
    return -1;
  }
  // strides should be less or equal than kernel size
  auto kernel_info = GetInput("Filter");
  if (kernel_info[0].shape[2] < strides_[0] ||
      kernel_info[0].shape[3] < strides_[1]) {
    Logger(verbose) << "Cannot handle the situation that kernel_size < strides"
                    << std::endl;
    return -1;
  }
  return 7;
}

void Conv2dMapper::Opset7() {
  auto kernel_info = GetInput("Filter");
  auto input_info = GetInput("Input");
  auto output_info = GetOutput("Output");
  auto input = helper_->AutoCast(input_info[0].name, input_info[0].dtype,
                                 P2ODataType::FP32);
  auto kernel = helper_->AutoCast(kernel_info[0].name, kernel_info[0].dtype,
                                  P2ODataType::FP32);
  auto node = helper_->MakeNode("Conv", {input, kernel});
  AddAttribute(node, "dilations", dilations_);
  std::vector<int64_t> kernel_shape = {kernel_info[0].shape[2],
                                       kernel_info[0].shape[3]};
  AddAttribute(node, "kernel_shape", kernel_shape);
  AddAttribute(node, "strides", strides_);
  AddAttribute(node, "group", groups_);
  if (padding_algorithm_ == "SAME") {
    std::string auto_pad = "SAME_UPPER";
    AddAttribute(node, "auto_pad", auto_pad);
  } else if (padding_algorithm_ == "VALID") {
    std::string auto_pad = "VALID";
    AddAttribute(node, "auto_pad", auto_pad);
  } else {
    AddAttribute(node, "pads", paddings_);
  }
  helper_->AutoCast(node->output(0), output_info[0].name, P2ODataType::FP32,
                    output_info[0].dtype);
}

}  // namespace paddle2onnx
