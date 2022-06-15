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

#include "paddle2onnx/mapper/tensor/scatter.h"

namespace paddle2onnx {
REGISTER_MAPPER(scatter, ScatterMapper)

int32_t ScatterMapper::GetMinOpset(bool verbose) {
  if (!overwrite_) {
    Error() << "overwrite = False not support yet." << std::endl;
    return -1;  // TODO(yeliang): overwrite can be False when opset version is
                // 16
  }
  Logger(verbose, 11) << RequireOpset(11) << std::endl;
  return 11;
}

void ScatterMapper::Opset11() {
  auto input_x_info = GetInput("X");
  auto input_ids_info = GetInput("Ids");
  auto input_updates_info = GetInput("Updates");
  auto output_info = GetOutput("Out");

  std::string ids_node = helper_->AutoCast(
      input_ids_info[0].name, input_ids_info[0].dtype, P2ODataType::INT64);

  std::vector<int64_t> shape_val = {input_ids_info[0].shape[0], 1};
  std::string shape_node =
      helper_->Constant(GetOnnxDtype(P2ODataType::INT64), shape_val);

  auto reshape_index_node =
      helper_->MakeNode("Reshape", {ids_node, shape_node});

  auto node = helper_->MakeNode(
      "ScatterND", {input_x_info[0].name, reshape_index_node->output(0),
                    input_updates_info[0].name},
      {output_info[0].name});
}

}  // namespace paddle2onnx
