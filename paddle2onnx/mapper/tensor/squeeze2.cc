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

#include "paddle2onnx/mapper/tensor/squeeze2.h"

namespace paddle2onnx {
REGISTER_MAPPER(squeeze2, Squeeze2Mapper)
REGISTER_PIR_MAPPER(squeeze, Squeeze2Mapper)

int32_t Squeeze2Mapper::GetMinOpsetVersion(bool verbose) {
  if (in_pir_mode) {
    // if (HasInput("axis")) {
    //   return 13;
    // }
    return 7;
  }

  if (IsAttrVar("axes")) {
    auto infos = GetAttrVar("axes");
    for (auto &info : infos) {
      if (!IsConstant(info)) {
        return 13;
      }
    }
  }
  return 7;
}

void Squeeze2Mapper::Opset7() {
  auto input_info = GetInput("X");
  auto output_info = GetOutput("Out");

  std::vector<int64_t> ret;
  ret.reserve(input_info[0].shape.size());
  for (auto i : input_info[0].shape) {
    if (i > 1) ret.push_back(i);
  }
  if (ret.size() == input_info[0].Rank()) {
    // All dimensions are > 1, nothing to squeeze
    helper_->MakeNode("Identity", {input_info[0].name}, {output_info[0].name});
  } else {
    bool with_axis = in_pir_mode ? HasInput("axis") : IsAttrVar("axes");
    if (helper_->GetOpsetVersion() >= 13 && with_axis) {
      auto axes_info = in_pir_mode ? GetInput("axis") : GetAttrVar("axes");

      // Check if we can get the axes values statically
      std::vector<int64_t> axes_values;
      bool axes_known = false;
      if (in_pir_mode) {
        axes_known = TryGetInputValue("axis", &axes_values);
      }

      // If axes are known, check if the dimensions at those axes are 1
      if (axes_known && !axes_values.empty()) {
        bool all_dims_not_one = true;
        for (auto axis : axes_values) {
          int64_t actual_axis = axis >= 0 ? axis : axis + input_info[0].Rank();
          if (actual_axis >= 0 && actual_axis < input_info[0].Rank()) {
            int64_t dim_size = input_info[0].shape[actual_axis];
            if (dim_size == 1 || dim_size == -1) {
              // -1 means dynamic, might be 1 at runtime
              all_dims_not_one = false;
              break;
            }
          }
        }
        if (all_dims_not_one) {
          // None of the dimensions to squeeze have size 1, use Identity
          helper_->MakeNode(
              "Identity", {input_info[0].name}, {output_info[0].name});
          return;
        }
      }

      std::string axes_name;
      if (axes_info.size() == 1U) {
        axes_name = helper_->AutoCast(
            axes_info[0].name, axes_info[0].dtype, P2ODataType::INT64);
      } else {
        axes_name = helper_->ConcatIndices(axes_info);
      }
      helper_->MakeNode(
          "Squeeze", {input_info[0].name, axes_name}, {output_info[0].name});
    } else {
      if (with_axis) {
        auto axes_info = in_pir_mode ? GetInput("axis") : GetAttrVar("axes");
        for (int64_t index = 0; index < axes_info.size(); index++) {
          std::vector<int64_t> temp;
          if (in_pir_mode) {
            TryGetInputValue("axis", &temp);
          } else {
            TryGetValue(axes_info[index], &temp);
          }
          for (auto &data : temp) {
            axes_.push_back(data);
          }
        }
      } else {
        GetAttr("axes", &axes_);
      }
      std::vector<int64_t> axes(axes_.begin(), axes_.end());
      for (size_t i = 0; i < axes.size(); ++i) {
        if (axes[i] < 0) {
          axes[i] += input_info[0].Rank();
        }
      }
      if (axes.size() > 0) {
        std::sort(axes.begin(), axes.end());
        helper_->Squeeze(input_info[0].name, output_info[0].name, axes);
      } else {
        helper_->Squeeze(input_info[0].name, output_info[0].name, {});
      }
    }
  }
}

}  // namespace paddle2onnx
