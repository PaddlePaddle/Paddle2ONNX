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

#include "paddle2onnx/mapper/nn/unfold.h"

#include <string>
#include <vector>

namespace paddle2onnx {
REGISTER_MAPPER(unfold, UnfoldMapper)
int32_t UnfoldMapper::GetMinOpsetVersion(bool verbose) {
    Logger(verbose, 11) << RequireOpset(11) << std::endl;
    return 11;
}
std::vector<int64_t> arange_(int64_t start, int64_t end, int64_t step) {
    std::vector<int64_t> result;
    for (int64_t i = start; i < end; i += step)
    {
        result.push_back(i);
    }
    return result;
}
std::vector<std::string> UnfoldMapper::_get_shape(std::string & x){
    std::string nchw = helper_->MakeNode("Shape", {x})->output(0);
    std::vector<std::string> nchw_vec = helper_->Split(nchw, {1,1,1,1}, 0);
    for (int i=0;i<nchw_vec.size();i++){
        nchw_vec[i] = helper_->Squeeze(nchw_vec[i], {}); 
    }
    return nchw_vec;
}
void UnfoldMapper::Opset11() {
    auto input_info = GetInput("X");
    auto output_info = GetOutput("Y");
    std::vector<std::string> nchw_vec = _get_shape(input_info[0].name);
    std::string input_batch = nchw_vec[0];
    std::string input_channel = nchw_vec[1];
    std::string input_h = nchw_vec[2];
    std::string input_w = nchw_vec[3];
    Assert(
        paddings_[0] == paddings_[2] && paddings_[1] == paddings_[3],
        "paddings[0] is not equal to paddings[2]!");

    int64_t padding_h = paddings_[0], padding_w = paddings_[2];
    int64_t kernel_h = kernel_sizes_[0], kernel_w = kernel_sizes_[1];
    int64_t stride_h = strides_[0], stride_w = strides_[1];
    int64_t dilation_h = dilations_[0], dilation_w = dilations_[1];

    std::string blocks_row_indices, blocks_col_indices, output_shape, padded_input;

    blocks_row_indices = _get_im2col_indices_along_dim(
        input_h, kernel_h, dilation_h, padding_h, stride_h);
    blocks_col_indices = _get_im2col_indices_along_dim(
        input_w, kernel_w, dilation_w, padding_w, stride_w);

    output_shape = _get_im2col_output_shape(input_batch, input_channel, kernel_h, kernel_w);
    padded_input = _get_im2col_padded_input(input_info[0].name, padding_h, padding_w);
    auto gather_node1 = helper_->MakeNode("Gather", {padded_input, blocks_row_indices});
    AddAttribute(gather_node1, "axis", (int64_t)2);
    auto gather_node2 = helper_->MakeNode("Gather", {gather_node1->output(0), blocks_col_indices});
    AddAttribute(gather_node2, "axis", (int64_t)4);

    auto transpose_node = helper_->MakeNode("Transpose", {gather_node2->output(0)});
    std::vector<int64_t> perm{0, 1, 2, 4, 3, 5};
    AddAttribute(transpose_node, "perm", perm); 
    helper_->MakeNode("Reshape", {transpose_node->output(0), output_shape}, {output_info[0].name});
}

std::string UnfoldMapper::_get_im2col_indices_along_dim(std::string intput_d,
          int64_t kernel_size_d, int64_t dialation_d, int64_t padding_d, int64_t stride_d){

    std::string blocks_d, blocks_d_indices, kernel_grid, kernel_mask, block_mask;
    blocks_d = helper_->MakeNode(
        "Add",
        {intput_d,
            helper_->Constant({}, ONNX_NAMESPACE::TensorProto::INT64, padding_d * 2)
        })->output(0);
    blocks_d = helper_->MakeNode(
        "Sub", 
        {blocks_d, 
          helper_->Constant({}, ONNX_NAMESPACE::TensorProto::INT64, dialation_d * (kernel_size_d - 1))
          })->output(0);
    blocks_d_indices = helper_->MakeNode(
        "Range",
        {helper_->Constant({}, ONNX_NAMESPACE::TensorProto::INT64, 0),
            blocks_d,
            helper_->Constant({}, ONNX_NAMESPACE::TensorProto::INT64, stride_d)
        })->output(0);
        std::vector<int64_t> kernel_grid_vec = arange_(0, kernel_size_d * dialation_d, dialation_d);
        kernel_grid = helper_->Constant({1, static_cast<int64_t>(kernel_grid_vec.size())}, ONNX_NAMESPACE::TensorProto::INT64, kernel_grid_vec);

        blocks_d_indices = helper_->Unsqueeze(blocks_d_indices, {0});
        kernel_mask = helper_->Reshape(kernel_grid, {-1, 1});

        block_mask = helper_->MakeNode("Add", {blocks_d_indices, kernel_mask})->output(0);
        return block_mask;
}
std::string UnfoldMapper::_get_im2col_padded_input(std::string & input_name, int64_t padding_h, int64_t padding_w){
        std::vector<int64_t> pad_constant{0, 0, padding_h, padding_w, 0, 0, padding_h, padding_w};
        std::string pad = helper_->Constant(
            ONNX_NAMESPACE::TensorProto::INT64,
            pad_constant
        );

        return helper_->MakeNode("Pad", {input_name, pad})->output(0);
}
std::string UnfoldMapper::_get_im2col_output_shape(std::string &batch_dim, std::string &channel_dim, int64_t kernel_h, int64_t kernel_w){
        std::string channel_unfolded = helper_->MakeNode(
            "Mul", 
            {channel_dim, 
                helper_->Constant({}, ONNX_NAMESPACE::TensorProto::INT64, kernel_h * kernel_w)
            }
        )->output(0);

        auto concat_node = helper_->MakeNode(
                "Concat",
                {helper_->Unsqueeze(batch_dim, {0}),
                    helper_->Unsqueeze(channel_unfolded, {0}),
                    helper_->Constant({1}, ONNX_NAMESPACE::TensorProto::INT64,-1)
                }
        );
        AddAttribute(concat_node, "axis", (int64_t) 0);
        return concat_node->output(0);
}
} // namespace paddle2onnx