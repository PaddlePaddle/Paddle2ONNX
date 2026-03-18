// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle2onnx/mapper/tensor/index_put.h"

namespace paddle2onnx {
REGISTER_MAPPER(index_put, IndexPutMapper)
REGISTER_PIR_MAPPER(index_put, IndexPutMapper)

int32_t IndexPutMapper::GetMinOpsetVersion(bool verbose) {
  // ScatterND requires opset 11, and reduction attribute requires opset 16
  if (accumulate_) {
    Logger(verbose, 16) << RequireOpset(16) << std::endl;
    return 16;
  }
  Logger(verbose, 11) << RequireOpset(11) << std::endl;
  return 11;
}

void IndexPutMapper::Opset11() {
  // Get inputs:
  // - x: the tensor to be updated
  // - indices: a vector of index tensors (one per dimension, can be boolean
  // masks)
  // - value: the values to put
  auto x_info = GetInput("x");
  auto indices_info = GetInput("indices");  // This is a vector of tensors
  auto value_info = GetInput("value");
  auto output_info = GetOutput("out");

  // Check if indices are boolean masks
  bool is_boolean_mask = false;
  if (indices_info.size() == 1 && indices_info[0].dtype == P2ODataType::BOOL) {
    is_boolean_mask = true;
  }

  if (is_boolean_mask) {
    // Boolean indexing: x[mask] = value
    // Use ONNX Where operator: out = Where(mask, value_broadcast, x)
    std::string mask = indices_info[0].name;

    // Cast value to match x's dtype if needed
    std::string value_name = value_info[0].name;
    if (value_info[0].dtype != x_info[0].dtype) {
      value_name = helper_->AutoCast(
          value_info[0].name, value_info[0].dtype, x_info[0].dtype);
    }

    // Expand value to match x's shape for broadcasting
    auto x_shape_node = helper_->MakeNode("Shape", {x_info[0].name});
    std::string value_broadcast =
        helper_->MakeNode("Expand", {value_name, x_shape_node->output(0)})
            ->output(0);

    if (accumulate_) {
      // When accumulate is true: x[mask] += value
      // out = Where(mask, x + value_broadcast, x)
      std::string add_result =
          helper_->MakeNode("Add", {x_info[0].name, value_broadcast})
              ->output(0);
      helper_->MakeNode(
          "Where", {mask, add_result, x_info[0].name}, {output_info[0].name});
    } else {
      // out = Where(mask, value_broadcast, x)
      helper_->MakeNode("Where",
                        {mask, value_broadcast, x_info[0].name},
                        {output_info[0].name});
    }
  } else {
    // Integer indexing: use ScatterND
    std::vector<std::string> indices_names;
    for (size_t i = 0; i < indices_info.size(); ++i) {
      // Cast indices to INT64 if needed
      std::string idx_name = helper_->AutoCast(
          indices_info[i].name, indices_info[i].dtype, P2ODataType::INT64);
      // Unsqueeze each index tensor to add a dimension at the end
      std::string axes_node = helper_->Constant(
          ONNX_NAMESPACE::TensorProto::INT64, std::vector<int64_t>{-1});
      auto unsqueeze_node =
          helper_->MakeNode("Unsqueeze", {idx_name, axes_node});
      indices_names.push_back(unsqueeze_node->output(0));
    }

    // Concat all indices along the last dimension
    std::string indices_concat;
    if (indices_names.size() == 1) {
      indices_concat = indices_names[0];
    } else {
      auto concat_node = helper_->MakeNode("Concat", indices_names);
      AddAttribute(concat_node, "axis", static_cast<int64_t>(-1));
      indices_concat = concat_node->output(0);
    }

    // Cast value to match x's dtype if needed
    std::string value_name = value_info[0].name;
    if (value_info[0].dtype != x_info[0].dtype) {
      value_name = helper_->AutoCast(
          value_info[0].name, value_info[0].dtype, x_info[0].dtype);
    }

    // For ScatterND, updates shape should be:
    // indices.shape[:-1] + data.shape[num_dims:]
    // where num_dims = indices.shape[-1] = number of index tensors

    // Get indices shape (without the last dim we added via Unsqueeze)
    auto indices_shape_node =
        helper_->MakeNode("Shape", {indices_info[0].name});

    // Get data shape and slice from num_dims onwards
    auto data_shape_node = helper_->MakeNode("Shape", {x_info[0].name});
    int64_t num_dims = static_cast<int64_t>(indices_info.size());
    auto start_const = helper_->Constant(ONNX_NAMESPACE::TensorProto::INT64,
                                         std::vector<int64_t>{num_dims});
    auto end_const = helper_->Constant(ONNX_NAMESPACE::TensorProto::INT64,
                                       std::vector<int64_t>{INT64_MAX});
    auto axes_const = helper_->Constant(ONNX_NAMESPACE::TensorProto::INT64,
                                        std::vector<int64_t>{0});
    auto data_shape_suffix = helper_
                                 ->MakeNode("Slice",
                                            {data_shape_node->output(0),
                                             start_const,
                                             end_const,
                                             axes_const})
                                 ->output(0);

    // Concat to get the target updates shape
    auto target_shape_node = helper_->MakeNode(
        "Concat", {indices_shape_node->output(0), data_shape_suffix});
    AddAttribute(target_shape_node, "axis", static_cast<int64_t>(0));

    // Expand value to match target shape
    value_name =
        helper_->MakeNode("Expand", {value_name, target_shape_node->output(0)})
            ->output(0);

    if (accumulate_) {
      auto shape_node = helper_->MakeNode("Shape", {x_info[0].name});
      std::string zeros_node =
          helper_->ConstOfShape(shape_node->output(0),
                                GetOnnxDtype(x_info[0].dtype),
                                static_cast<float>(0));

      auto scatter_node = helper_->MakeNode(
          "ScatterND", {zeros_node, indices_concat, value_name});
      AddAttribute(scatter_node, "reduction", std::string("add"));

      helper_->MakeNode("Add",
                        {x_info[0].name, scatter_node->output(0)},
                        {output_info[0].name});
    } else {
      helper_->MakeNode("ScatterND",
                        {x_info[0].name, indices_concat, value_name},
                        {output_info[0].name});
    }
  }
}

}  // namespace paddle2onnx
