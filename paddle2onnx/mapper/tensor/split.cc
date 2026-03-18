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

#include "paddle2onnx/mapper/tensor/split.h"

namespace paddle2onnx {
REGISTER_MAPPER(split, SplitMapper)
REGISTER_PIR_MAPPER(split, SplitMapper)
REGISTER_PIR_MAPPER(split_with_num, SplitMapper)

int32_t SplitMapper::GetMinOpsetVersion(bool verbose) {
  int64_t axis = axis_;
  if (HasInput("axis") || HasInput("AxisTensor")) {
    if (in_pir_mode) {
      double value = 0;
      std::string attr_name = "AxisTensor";
      if (HasInput("axis")) {
        attr_name = "axis";
      }
      std::string err_msg = "While " + attr_name +
                            " as the input and it's not a constant tensor, the "
                            "conversion is not supported yet.";
      Assert(TryGetInputValue(attr_name, &value), err_msg);
      axis = (int64_t)value;
    } else {
      std::vector<int64_t> value;
      if (!TryGetInputValue("AxisTensor", &value)) {
        Error() << "While AxisTensor as the input and it's not a constant "
                   "tensor, the conversion is not supported yet."
                << std::endl;
        return -1;
      }
      axis = value[0];
    }
  }

  if (HasInput("sections")) {
    if (!TryGetInputValue("sections", &sections_)) {
      Logger(verbose, 13) << "While input sections is not a constant tensor, "
                          << RequireOpset(13) << std::endl;
      return 13;
    }
  }
  if (HasInput("SectionsTensorList")) {
    Logger(verbose, 13) << "While has input SectionsTensorList, "
                        << RequireOpset(13) << std::endl;
    return 13;
  }

  for (size_t i = 0; i < sections_.size(); ++i) {
    if (sections_[i] < 0) {
      auto info = GetInput("X");
      if (info[0].shape[axis] < 0) {
        Error() << "Cannot convert split op, while there's -1 in sections and "
                   "cannot be infered by input shape."
                << std::endl;
        return -1;
      }
    }
  }
  return 7;
}

int64_t SplitMapper::GetAxis(int64_t rank) {
  int64_t axis = axis_;  // Old IR
  if (HasInput("axis") || HasInput("AxisTensor")) {
    if (in_pir_mode) {
      double value = 0;
      std::string attr_name = "AxisTensor";
      if (HasInput("axis")) attr_name = "axis";
      TryGetInputValue(attr_name, &value);
      axis = (int64_t)value;
    } else {
      std::vector<int64_t> value;
      Assert(TryGetInputValue("AxisTensor", &value),
             "[Paddle2ONNX](split) Cannot get constant value from AxisTensor.");
      axis = value[0];
    }
  }
  if (axis < 0) {
    axis += rank;
  }
  return axis;
}

std::string SplitMapper::GetSections(int64_t dimension) {
  std::string splits = "";
  if (HasInput("SectionsTensorList")) {
    auto info = GetInput("SectionsTensorList");
    splits = helper_->ConcatIndices(info);
  } else if (sections_.size() > 0 ||
             (HasInput("sections") &&
              TryGetInputValue("sections", &sections_))) {
    ProcessSections(dimension);
    splits = helper_->Constant(ONNX_NAMESPACE::TensorProto::INT64, sections_);
  } else if (HasInput("sections")) {
    auto info = GetInput("sections");
    splits = helper_->ConcatIndices(info);
  } else if (HasAttr("num")) {
    GetAttr("num", &num_);
    Assert(dimension > 0,
           "Cannot convert split op, due to target dimension is dynamic.");
    int64_t each_part_size = dimension / num_;
    sections_ = std::vector<int64_t>(num_, each_part_size);
    sections_[num_ - 1] += dimension % num_;
  }
  return splits;
}

void SplitMapper::ProcessSections(int64_t dimension) {
  int sum_of_known_dim = 0;
  int cnt_of_unknown_dim = 0;
  for (size_t i = 0; i < sections_.size(); ++i) {
    if (sections_[i] > 0) {
      sum_of_known_dim += sections_[i];
    } else {
      cnt_of_unknown_dim++;
    }
  }
  Assert(cnt_of_unknown_dim <= 1,
         "Cannot convert split op, while there's more than 1 -1 in sections.");
  for (size_t i = 0; i < sections_.size(); ++i) {
    if (sections_[i] < 0) {
      Assert(dimension > 0,
             "Cannot convert split op, while there's -1 in sections and cannot "
             "be infered by input shape.");
      sections_[i] = dimension - sum_of_known_dim;
    }
  }
}

void SplitMapper::Opset7() {
  std::vector<TensorInfo> input_info;
  std::vector<TensorInfo> output_info;
  if (HasInput("X")) {
    input_info = GetInput("X");
  } else {
    input_info = GetInput("x");
  }
  if (HasOutput("Out")) {
    output_info = GetOutput("Out");
  } else {
    output_info = GetOutput("out");
  }
  int64_t axis = GetAxis(input_info[0].Rank());
  if (HasInput("sections")) {
    TryGetInputValue("sections", &sections_);
    Assert(sections_.size() > 0,
           "While sections is empty, the conversion is not supported yet.");
  }

  if (sections_.size() > 0) {
    ProcessSections(input_info[0].shape[axis]);
  } else if (HasAttr("num")) {
    GetAttr("num", &num_);
    int64_t each_part_size = input_info[0].shape[axis] / num_;
    sections_ = std::vector<int64_t>(num_, each_part_size);
    sections_[num_ - 1] += input_info[0].shape[axis] % num_;
  } else {
    Assert(false, "Split op must have sections or num attribute.");
  }

  std::vector<std::string> output_names(output_info.size());
  for (size_t i = 0; i < output_info.size(); ++i) {
    output_names[i] = output_info[i].name;
  }

  helper_->Split(input_info[0].name, output_names, sections_, axis);
}

void SplitMapper::Opset13() {
  std::vector<TensorInfo> input_info;
  std::vector<TensorInfo> output_info;
  if (HasInput("X")) {
    input_info = GetInput("X");
  } else {
    input_info = GetInput("x");
  }
  if (HasOutput("Out")) {
    output_info = GetOutput("Out");
  } else {
    output_info = GetOutput("out");
  }

  int64_t axis = GetAxis(input_info[0].Rank());
  std::string splits = GetSections(input_info[0].shape[axis]);

  std::vector<std::string> output_names(output_info.size());
  for (size_t i = 0; i < output_info.size(); ++i) {
    output_names[i] = output_info[i].name;
  }
  if (splits != "") {
    auto node =
        helper_->MakeNode("Split", {input_info[0].name, splits}, output_names);
    AddAttribute(node, "axis", axis);
  } else {
    if (HasAttr("num")) {
      helper_->Split(input_info[0].name, output_names, sections_, axis);
    } else {
      auto node =
          helper_->MakeNode("Split", {input_info[0].name}, output_names);
      AddAttribute(node, "axis", axis);
    }
  }
}

void SplitMapper::Opset18() {
  std::vector<TensorInfo> input_info;
  std::vector<TensorInfo> output_info;
  if (HasInput("X")) {
    input_info = GetInput("X");
  } else {
    input_info = GetInput("x");
  }
  if (HasOutput("Out")) {
    output_info = GetOutput("Out");
  } else {
    output_info = GetOutput("out");
  }

  int64_t axis = GetAxis(input_info[0].Rank());
  std::string splits = GetSections(input_info[0].shape[axis]);

  std::vector<std::string> output_names(output_info.size());
  for (size_t i = 0; i < output_info.size(); ++i) {
    output_names[i] = output_info[i].name;
  }
  if (splits != "") {
    auto node =
        helper_->MakeNode("Split", {input_info[0].name, splits}, output_names);
    AddAttribute(node, "axis", axis);
  } else {
    int64_t num = input_info[0].shape[axis];  // default
    if (HasAttr("num")) {
      GetAttr("num", &num);
    }
    auto node = helper_->MakeNode("Split", {input_info[0].name}, output_names);
    AddAttribute(node, "axis", axis);
    AddAttribute(node, "num_outputs", num);
  }
}

}  // namespace paddle2onnx
