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

#include "paddle2onnx/mapper/nn/pool.h"
#include <math.h>
#include <algorithm>
#include <string>
#include <vector>

namespace paddle2onnx {
REGISTER_MAPPER(pool2d, Pool2dMapper)

bool Pool2dMapper::IsSameSpan(const int64_t& in_size, const int64_t& out_size) {
  std::vector<int64_t> spans;
  spans.reserve(out_size);
  for (auto i = 0; i < out_size; ++i) {
    int64_t start = std::floor(i * (in_size / out_size));
    int64_t end = std::ceil((i + 1) * (in_size / out_size));
    spans.push_back(end - start);
  }
  std::sort(spans.begin(), spans.end());
  return spans[0] == spans[spans.size() - 1];
}

int32_t Pool2dMapper::GetMinOpset(bool verbose) {
  // NHWC is not supported
  if (data_format_ == "NHWC") {
    if (verbose) {
      std::cerr << "[ERROR] Cannot support NHWC format for operator pool2d."
                << std::endl;
    }
    return -1;
  }
  auto op = parser_->GetOpDesc(block_idx_, op_idx_);
  std::vector<TensorInfo> input_info =
      parser_->GetOpInput(block_idx_, op_idx_, "X");
  std::vector<TensorInfo> output_info =
      parser_->GetOpOutput(block_idx_, op_idx_, "Out");
  bool adaptive;
  parser_->GetOpAttr(op, "adaptive", &adaptive);
  if (adaptive) {
    if (!parser_->IsStaticShape(input_info)) {
      if (verbose) {
        std::cerr << "[ERROR] Adaptive only support static input shape for "
                     "operator pool2d."
                  << std::endl;
      }
      return -1;
    }
    int64_t input_h = input_info[0].shape[2];
    int64_t input_w = input_info[0].shape[3];
    int64_t output_h = output_info[0].shape[2];
    int64_t output_w = output_info[0].shape[3];
    if (!IsSameSpan(input_h, output_h) || !IsSameSpan(input_w, output_w)) {
      if (verbose) {
        std::cerr << "[ERROR] Cannot convert adaptive pool with input_size: "
                  << input_h << " " << input_h << " output_size: " << output_h
                  << " " << output_w << std::endl;
      }
      return -1;
    }
  }

  std::string pooling_type;
  parser_->GetOpAttr(op, "pooling_type", &pooling_type);
  auto iter = op_mapper_.find(pooling_type);
  if (op_mapper_.end() == iter) {
    if (verbose) {
      std::cerr << "[ERROR] Cannot find " + pooling_type +
                       " in pool op_mapper.."
                << std::endl;
    }
    return -1;
  }

  bool ceil_mod = false;
  parser_->GetOpAttr(op, "ceil_mode", &ceil_mod);
  if (ceil_mod) {
    return 10;
  }
  return 7;
}

void Pool2dMapper::Opset7(OnnxHelper* helper) {
  auto op = parser_->GetOpDesc(block_idx_, op_idx_);
  std::vector<TensorInfo> input_info =
      parser_->GetOpInput(block_idx_, op_idx_, "X");
  std::vector<TensorInfo> output_info =
      parser_->GetOpOutput(block_idx_, op_idx_, "Out");

  bool global_pooling = false;
  parser_->GetOpAttr(op, "global_pooling", &global_pooling);

  bool adaptive = false;
  std::vector<int64_t> ksize;
  parser_->GetOpAttr(op, "adaptive", &adaptive);
  parser_->GetOpAttr(op, "ksize", &ksize);
  bool k_is_one = true;
  for (auto i : ksize) {
    if (i != 1) k_is_one = false;
  }

  std::string pooling_type;
  parser_->GetOpAttr(op, "pooling_type", &pooling_type);
  if (global_pooling || (adaptive && k_is_one)) {
    auto iter = op_mapper_.find(pooling_type);
    auto node = helper->MakeNode(iter->second[1], {input_info[0].name},
                                 {output_info[0].name});
  } else if (adaptive) {
    int64_t input_h = input_info[0].shape[2];
    int64_t input_w = input_info[0].shape[3];
    int64_t output_h = output_info[0].shape[2];
    int64_t output_w = output_info[0].shape[3];
    int64_t stride_h = std::floor(input_h / output_h);
    int64_t stride_w = std::floor(input_w / output_w);
    int64_t kernel_h = input_h - (output_h - 1) * stride_h;
    int64_t kernel_w = input_w - (output_w - 1) * stride_w;
    auto iter = op_mapper_.find(pooling_type);
    auto node = helper->MakeNode(iter->second[0], {input_info[0].name},
                                 {output_info[0].name});
    std::vector<int64_t> kernel_size = {kernel_h, kernel_w};
    AddAttribute(node, "kernel_shape", kernel_size);
    std::vector<int64_t> strides = {stride_h, stride_w};
    AddAttribute(node, "strides", strides);

    bool ceil_mod = false;
    parser_->GetOpAttr(op, "ceil_mode", &ceil_mod);

    if (helper->GetOpsetVersion() > 10) {
      AddAttribute(node, "ceil_mode", ceil_mod);
    }

    std::string padding_algorithm;
    parser_->GetOpAttr(op, "padding_algorithm", &padding_algorithm);
    std::string auto_pad = "NOTSET";
    if (padding_algorithm == "SAME") {
      auto_pad = "SAME_UPPER";
    } else if (padding_algorithm == "VALID") {
      auto_pad = "VALID";
    }
    AddAttribute(node, "auto_pad", auto_pad);
    if (pooling_type == "avg") {
      bool exclusive = false;
      parser_->GetOpAttr(op, "exclusive", &exclusive);
      exclusive = !exclusive;
      AddAttribute(node, "count_include_pad", exclusive);
    }
  } else {
    std::vector<int64_t> input_shape = input_info[0].shape;
    std::vector<int64_t> k_size;
    parser_->GetOpAttr(op, "ksize", &k_size);
    std::vector<int64_t> pads;
    parser_->GetOpAttr(op, "paddings", &pads);
    std::vector<int64_t> strides;
    parser_->GetOpAttr(op, "strides", &strides);
    if (pads.size() == 2) {
      pads.push_back(pads[0]);
      pads.push_back(pads[1]);
    } else if (pads.size() == 4) {
      std::vector<int64_t> index = {0, 2, 1, 3};
      std::vector<int64_t> copy = pads;
      for (auto i = 0; i < index.size(); ++i) {
        pads[i] = copy[index[i]];
      }
    }
    if (input_shape[2] > 0 && input_shape[2] + pads[0] < k_size[0]) {
      k_size[0] = input_shape[2] + pads[0];
    }
    if (input_shape[3] > 0 && input_shape[3] + pads[1] < k_size[1]) {
      k_size[1] = input_shape[3] + pads[1];
    }

    int64_t max_ksize = *std::max_element(std::begin(k_size), std::end(k_size));
    int64_t max_pads = *std::max_element(std::begin(pads), std::end(pads));
    std::string input_x = input_info[0].name;
    if (max_ksize <= max_pads) {
      std::vector<int64_t> onnx_paddings = {0, 0, pads[0], pads[1],
                                            0, 0, pads[2], pads[3]};
      std::vector<std::string> inputs_names = {input_x};
      if (helper->GetOpsetVersion() >= 11) {
        std::string paddings_node =
            helper->Constant(GetOnnxDtype(P2ODataType::INT64), onnx_paddings);
        inputs_names.push_back(paddings_node);
        std::vector<float> val = {0.0};
        std::string val_node =
            helper->Constant(GetOnnxDtype(P2ODataType::FP32), val);
        inputs_names.push_back(val_node);
      }
      auto node = helper->MakeNode("Pad", inputs_names);
      std::string mode = "constant";
      AddAttribute(node, "mode", mode);
      if (helper->GetOpsetVersion() < 11) {
        AddAttribute(node, "pads", onnx_paddings);
        float val = 0.0;
        AddAttribute(node, "value", val);
      }
      input_x = node->output(0);
      pads.clear();
      pads.resize(4, 0);
    }

    auto iter = op_mapper_.find(pooling_type);
    auto node =
        helper->MakeNode(iter->second[0], {input_x}, {output_info[0].name});
    AddAttribute(node, "kernel_shape", k_size);
    AddAttribute(node, "strides", strides);
    std::string padding_algorithm;
    parser_->GetOpAttr(op, "padding_algorithm", &padding_algorithm);
    std::string auto_pad = "NOTSET";
    if (padding_algorithm == "SAME") {
      auto_pad = "SAME_UPPER";
    } else if (padding_algorithm == "VALID") {
      auto_pad = "VALID";
    }
    AddAttribute(node, "auto_pad", auto_pad);
    AddAttribute(node, "pads", pads);
    bool ceil_mod = false;
    parser_->GetOpAttr(op, "ceil_mode", &ceil_mod);
    if (helper->GetOpsetVersion() >= 10) {
      AddAttribute(node, "ceil_mode", ceil_mod);
    }
    if (pooling_type == "avg") {
      bool exclusive = false;
      parser_->GetOpAttr(op, "exclusive", &exclusive);
      exclusive = !exclusive;
      AddAttribute(node, "count_include_pad", exclusive);
    }
  }
}

}  // namespace paddle2onnx
