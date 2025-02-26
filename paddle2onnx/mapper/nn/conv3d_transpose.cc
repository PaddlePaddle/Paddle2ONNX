// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle2onnx/mapper/nn/conv3d_transpose.h"

namespace paddle2onnx {
REGISTER_PIR_MAPPER(conv3d_transpose, Conv3dTransposeMapper)

int32_t Conv3dTransposeMapper::GetMinOpsetVersion(bool verbose) {
  if (data_format_ != "NCHW" && data_format_ != "NHWC") {
    Error() << "[ERROR] only support NCDHW or NDHWC format for "
               "conv3d_transpose "
            << std::endl;
    return -1;
  }
  return 7;
}

void Conv3dTransposeMapper::Opset7() {
  auto kernel_info = GetInput("filter");
  auto input_info = GetInput("x");
  auto output_info = GetOutput("out");

  auto input = helper_->AutoCast(
      input_info[0].name, input_info[0].dtype, P2ODataType::FP32);
  if (data_format_ == "NHWC") {
    input = helper_->Transpose(input, {0, 4, 1, 2, 3});  // NDHWC -> NCDHW
  }

  auto kernel = helper_->AutoCast(
      kernel_info[0].name, kernel_info[0].dtype, P2ODataType::FP32);

  auto node = helper_->MakeNode("ConvTranspose", {input, kernel});

  std::vector<int64_t> kernel_shape = {kernel_info[0].shape[2],
                                       kernel_info[0].shape[3],
                                       kernel_info[0].shape[4]};
  AddAttribute(node, "kernel_shape", kernel_shape);

  AddAttribute(node, "dilations", dilations_);
  AddAttribute(node, "strides", strides_);
  AddAttribute(node, "group", groups_);

  if (padding_algorithm_ == "SAME") {
    AddAttribute(node, "auto_pad", "SAME_UPPER");
  } else if (padding_algorithm_ == "VALID") {
    AddAttribute(node, "auto_pad", "VALID");
  } else {
    std::vector<int64_t> paddings;
    if (paddings_.size() == 3) {
      paddings.insert(paddings.begin(), paddings_.begin(), paddings_.end());
      paddings.insert(paddings.begin(), paddings_.begin(), paddings_.end());
    } else {
      std::vector<int64_t> index = {0, 2, 4, 1, 3, 5};
      for (auto& i : index) {
        paddings.push_back(paddings_[i]);
      }
    }
    AddAttribute(node, "pads", paddings);
  }

  if (!output_padding_.empty()) {
    AddAttribute(node, "output_padding", output_padding_);
  }

  auto output = node->output(0);
  if (data_format_ == "NHWC") {
    output = helper_->Transpose(output, {0, 2, 3, 4, 1});  // NCDHW -> NDHWC
  }
  helper_->AutoCast(
      output, output_info[0].name, P2ODataType::FP32, output_info[0].dtype);
}
}  // namespace paddle2onnx
