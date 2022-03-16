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

#include "paddle2onnx/mapper/tensor/slice.h"
#include <iostream>
#include <string>
#include <vector>

namespace paddle2onnx {
REGISTER_MAPPER(slice, SliceMapper)

int32_t SliceMapper::GetMinOpset(bool verbose) {
  std::string starts_node;
  std::vector<int64_t> starts;
  bool starts_is_tensor =
      GetNodeAttrValue("starts", "StartsTensor", "StartsTensorList", &starts,
                       &starts_node, true);

  std::string ends_node;
  std::vector<int64_t> ends;
  bool ends_is_tensor = GetNodeAttrValue("ends", "EndsTensor", "EndsTensorList",
                                         &ends, &ends_node, true);
  if (starts_is_tensor || ends_is_tensor) {
    return 10;
  }
  return 7;
}

std::vector<int64_t> SliceMapper::DecreaseAxis() {
  auto op = parser_->GetOpDesc(block_idx_, op_idx_);
  std::vector<int64_t> decrease_axis;
  bool has_attr = parser_->OpHasAttr(op, "decrease_axis");
  if (has_attr) {
    parser_->GetOpAttr(op, "decrease_axis", &decrease_axis);
    std::vector<TensorInfo> input_info =
        parser_->GetOpInput(block_idx_, op_idx_, "Input");
    std::vector<TensorInfo> output_info =
        parser_->GetOpOutput(block_idx_, op_idx_, "Out");
    if (output_info[0].shape.size() == 1 && output_info[0].shape[0] == 0) {
      return decrease_axis;
    }
    if (input_info[0].shape.size() > output_info[0].shape.size()) {
      return decrease_axis;
    }
    return {};
  }
  return decrease_axis;
}

bool SliceMapper::GetNodeAttrValue(
    const std::string &attr_name, const std::string &attr_tensor_name,
    const std::string &attr_tensor_list_name, std::vector<int64_t> *val,
    std::string *val_tensor, const bool &return_list, OnnxHelper *helper) {
  bool has_attr_tensor =
      parser_->OpHasInput(block_idx_, op_idx_, attr_tensor_name);
  bool has_attr_tensor_list =
      parser_->OpHasInput(block_idx_, op_idx_, attr_tensor_list_name);
  if (has_attr_tensor) {
    std::vector<TensorInfo> input_tensor_info =
        parser_->GetOpInput(block_idx_, op_idx_, attr_tensor_name);
    if (return_list) {
      std::vector<int64_t> index =
          parser_->GetBlockOpIdx(input_tensor_info[0].name);
      Weight value;
      bool found_value =
          parser_->GetValueFromTensor(index[0], index[1], &value);
      if (found_value) {
        value.get(val);
      }
      return !found_value;
    } else {
      *val_tensor = input_tensor_info[0].name;
      return true;
    }
  }
  if (has_attr_tensor_list) {
    if (helper == nullptr) {
      return true;
    }
    std::vector<TensorInfo> input_info =
        parser_->GetOpInput(block_idx_, op_idx_, attr_tensor_list_name);
    int32_t casted_dtype;
    std::vector<std::string> casted_names =
        helper->DtypeAlignment(input_info, &casted_dtype);
    auto node = helper->MakeNode("Concat", casted_names);
    int64_t axis = 0;
    AddAttribute(node, "axis", axis);
    *val_tensor = node->output(0);
    return true;
  }
  auto op = parser_->GetOpDesc(block_idx_, op_idx_);
  bool has_attr = parser_->OpHasAttr(op, attr_name);
  if (has_attr) {
    parser_->GetOpAttr(op, attr_name, val);
  }
  return false;
}

void SliceMapper::Opset7(OnnxHelper *helper) {
  auto op = parser_->GetOpDesc(block_idx_, op_idx_);
  std::vector<TensorInfo> input_info =
      parser_->GetOpInput(block_idx_, op_idx_, "Input");
  std::vector<TensorInfo> output_info =
      parser_->GetOpOutput(block_idx_, op_idx_, "Out");

  std::string starts_node;
  std::vector<int64_t> starts;
  bool starts_is_tensor =
      GetNodeAttrValue("starts", "StartsTensor", "StartsTensorList", &starts,
                       &starts_node, true, helper);

  std::string ends_node;
  std::vector<int64_t> ends;
  bool ends_is_tensor = GetNodeAttrValue("ends", "EndsTensor", "EndsTensorList",
                                         &ends, &ends_node, true, helper);

  std::vector<int64_t> decrease_axis = DecreaseAxis();
  if (decrease_axis.empty()) {
    helper->Slice(input_info[0].name, output_info[0].name, axes_, starts, ends);
  } else {
    std::string node = helper->Slice(input_info[0].name, axes_, starts, ends);
    helper->Squeeze(node, output_info[0].name, decrease_axis);
  }
}

void SliceMapper::Opset10(OnnxHelper *helper) {
  auto op = parser_->GetOpDesc(block_idx_, op_idx_);
  std::vector<TensorInfo> input_info =
      parser_->GetOpInput(block_idx_, op_idx_, "Input");
  std::vector<TensorInfo> output_info =
      parser_->GetOpOutput(block_idx_, op_idx_, "Out");

  std::string strides_node =
      helper->MakeConstant(ONNX_NAMESPACE::TensorProto::INT64, strides_)
          ->output(0);

  std::string starts_node;
  std::vector<int64_t> starts;
  bool starts_is_tensor =
      GetNodeAttrValue("starts", "StartsTensor", "StartsTensorList", &starts,
                       &starts_node, false, helper);
  if (!starts_is_tensor) {
    starts_node =
        helper->MakeConstant(ONNX_NAMESPACE::TensorProto::INT64, starts)
            ->output(0);
  }

  std::string ends_node;
  std::vector<int64_t> ends;
  bool ends_is_tensor = GetNodeAttrValue("ends", "EndsTensor", "EndsTensorList",
                                         &ends, &ends_node, false, helper);
  if (!ends_is_tensor) {
    ends_node = helper->MakeConstant(ONNX_NAMESPACE::TensorProto::INT64, ends)
                    ->output(0);
  }

  std::string axes_node =
      helper->MakeConstant(ONNX_NAMESPACE::TensorProto::INT64, axes_)
          ->output(0);

  std::vector<int64_t> decrease_axis = DecreaseAxis();
  if (decrease_axis.empty()) {
    helper->MakeNode("Slice", {input_info[0].name, starts_node, ends_node,
                               axes_node, strides_node},
                     {output_info[0].name});
  } else {
    auto node = helper->MakeNode("Slice", {input_info[0].name, starts_node,
                                           ends_node, axes_node, strides_node});
    helper->Squeeze(node->output(0), output_info[0].name, decrease_axis);
  }
}

}  // namespace paddle2onnx
