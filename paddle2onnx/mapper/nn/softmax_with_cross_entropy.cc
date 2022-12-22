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

#include "paddle2onnx/mapper/nn/softmax_with_cross_entropy.h"

namespace paddle2onnx {
REGISTER_MAPPER(softmax_with_cross_entropy, SoftmaxCrossEntropyLossMapper)

int32_t SoftmaxCrossEntropyLossMapper::GetMinOpset(bool verbose) {
  if (soft_label_) {
    Error() << "SoftmaxCrossEntropyLoss in onnx not support soft label."
            << std::endl;
    return -1;
  }
  auto input_info = GetInput("Logits");
  std::vector<int64_t> input_shape = input_info[0].shape;
  if (input_shape.size() < 2) {
    Error() << "SoftmaxCrossEntropyLoss in onnx not support 1D logits."
            << std::endl;
    return -1;
  }
  Logger(verbose, 12) << RequireOpset(12) << std::endl;
  return 12;
}

void SoftmaxCrossEntropyLossMapper::Opset12() {
  auto logits = GetInput("Logits");
  auto labels = GetInput("Label");

  auto loss = GetOutput("Loss");
  auto softmax = GetOutput("Softmax");
  std::vector<int64_t> logits_shape = logits[0].shape;
  size_t dim = logits_shape.size();
  if (axis_ < 0) {
    axis_ += dim;
  } else if (axis_ == 1) {
    auto squeeze_node =
        // helper_->Squeeze(labels[0].name, std::vector<int64_t>(1, axis_));
        helper_->Squeeze(labels[0].name, std::vector<int64_t>(1, axis_));
    auto node = helper_->MakeNode("SoftmaxCrossEntropyLoss",
                                  {logits[0].name, squeeze_node},
                                  {loss[0].name, softmax[0].name});
    AddAttribute(node, "ignore_index", ignore_index_);
    AddAttribute(node, "reduction", "none");
    auto loss_node = helper_->Unsqueeze(node->output(0), loss[0].name, {axis_});
    // onnx output is log(softmax), but paddle output is softmax
    helper_->MakeNode("Exp", {node->output(1)}, {softmax[0].name});
  } else {
    std::vector<int64_t> perm = Arange(0, dim);
    perm[1] = axis_;
    perm[axis_] = 1;
    auto transpose_logits = helper_->MakeNode("Transpose", {logits[0].name});
    AddAttribute(transpose_logits, "perm", perm);
    auto transpose_labels = helper_->MakeNode("Transpose", {labels[0].name});
    AddAttribute(transpose_labels, "perm", perm);
    auto squeeze_labels =
        helper_->Squeeze(transpose_labels->name(), std::vector<int64_t>(1, 1));
    auto node = helper_->MakeNode("SoftmaxCrossEntropyLoss",
                                  {transpose_logits->name(), squeeze_labels},
                                  {loss[0].name, softmax[0].name});
    AddAttribute(node, "ignore_index", ignore_index_);
    AddAttribute(node, "reduction", "none");
    auto unsqueeze_node =
        helper_->Unsqueeze(node->output(0), std::vector<int64_t>(1, 1));
    auto revert_transpose_logits =
        helper_->MakeNode("Transpose", {unsqueeze_node}, {loss[0].name});
    AddAttribute(revert_transpose_logits, "perm", perm);
    auto softmax_node = helper_->MakeNode("Transpose", {node->output(1)});
    AddAttribute(softmax_node, "perm", perm);
    // onnx output is log(softmax), but paddle output is softmax
    helper_->MakeNode("Exp", {softmax_node->name()}, {softmax[0].name});
  }
}
}  // namespace paddle2onnx
