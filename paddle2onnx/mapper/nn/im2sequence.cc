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

#include "paddle2onnx/mapper/nn/im2sequence.h"

#include <string>
#include <vector>

namespace paddle2onnx
{
    REGISTER_MAPPER(im2sequence, Im2SequenceMapper)

    int32_t Im2SequenceMapper::GetMinOpset(bool verbose)
    {
        return 7;
    }

    std::vector<int64_t> CalculateNewShape(const std::vector<int64_t> &input_shape,
                                           const std::vector<int64_t> &kernels,
                                           const std::vector<int64_t> &strides,
                                           const std::vector<int64_t> &paddings)
    {
        int64_t N = input_shape[0];
        int64_t C = input_shape[1];
        int64_t H = input_shape[2];
        int64_t W = input_shape[3];

        int64_t kernel_h = kernels[0];
        int64_t kernel_w = kernels[1];
        int64_t stride_h = strides[0];
        int64_t stride_w = strides[1];
        int64_t pad_top = paddings[0];
        int64_t pad_bottom = paddings[1];
        int64_t pad_left = paddings[2];
        int64_t pad_right = paddings[3];

        int64_t out_h = (H + pad_top + pad_bottom - kernel_h) / stride_h + 1;
        int64_t out_w = (W + pad_left + pad_right - kernel_w) / stride_w + 1;

        std::vector<int64_t> new_shape = {N, C, out_h * kernel_h, out_w * kernel_w};

        return new_shape;
    }

    void Im2SequenceMapper::Opset7()
    {
        auto input_info = GetInput("X");
        auto output_info = GetOutput("Out");

        std::vector<int64_t> kernels = kernels_;
        std::vector<int64_t> strides = strides_;
        std::vector<int64_t> paddings = paddings_;

        std::vector<int64_t> new_shape = CalculateNewShape(input_info[0].shape, kernels_, strides_, paddings_);
        std::vector<int64_t> starts = {0, 0}; 
        std::vector<int64_t> ends = {-1, -1}; 
        std::vector<int64_t> axes = {2, 3};   
        std::vector<int64_t> steps = {1, 1};  

        std::string reshaped_input = helper_->Reshape(input_info[0].name, new_shape);

        std::vector<int64_t> starts_i64(starts.begin(), starts.end());
        std::vector<int64_t> ends_i64(ends.begin(), ends.end());
        std::vector<int64_t> axes_i64(axes.begin(), axes.end());
        std::vector<int64_t> steps_i64(steps.begin(), steps.end());
        std::string sliced_input = helper_->Slice(reshaped_input, starts_i64, ends_i64, axes_i64);
        helper_->MakeNode("Identity", {sliced_input}, {output_info[0].name});
    }

}