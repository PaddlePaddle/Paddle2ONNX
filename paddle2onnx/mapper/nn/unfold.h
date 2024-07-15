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

#pragma once
#include <string>
#include <vector>

#include "paddle2onnx/mapper/mapper.h"

namespace paddle2onnx {

class UnfoldMapper : public Mapper {
 public:
  UnfoldMapper(const PaddleParser& p, OnnxHelper* helper, int64_t block_id,
               int64_t op_id)
      : Mapper(p, helper, block_id, op_id) {
    GetAttr("dilations", &dilations_);
    GetAttr("strides", &strides_);
    GetAttr("paddings", &paddings_);
    GetAttr("kernel_sizes", &kernel_sizes_);
  }

  int32_t GetMinOpsetVersion(bool verbose = false);
  void Opset11();
  std::string _get_im2col_indices_along_dim(std::string intput_d, int64_t kernel_size_d, int64_t dialation_d,  int64_t padding_d, int64_t stride_d);
  std::string _get_im2col_output_shape(std::string & batch_dim, std::string & channel_dim, int64_t kernel_h, int64_t kernel_w);
  std::string _get_im2col_padded_input(std::string & input_name, int64_t padding_h, int64_t padding_w);
  std::vector<std::string> _get_shape(std::string & x);
 private:
  std::vector<int64_t> dilations_;
  std::vector<int64_t> strides_;
  std::vector<int64_t> paddings_;
  std::vector<int64_t> kernel_sizes_;
};

}  // namespace paddle2onnx
