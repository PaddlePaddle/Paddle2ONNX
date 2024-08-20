// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle2onnx/parser/tensor_utils.h"

#include <cstddef>
#include <cstdint>

#include <algorithm>
#include <cassert>
#include <cstring>
#include <numeric>
#include <string>
#include <type_traits>
#include <vector>

namespace paddle2onnx {
enum P2ODataType {
  BOOL,
  INT16,
  INT32,
  INT64,
  FP16,
  FP32,
  FP64,
  X7,
  X8,
  X9,
  X10,
  X11,
  X12,
  X13,
  X14,
  X15,
  X16,
  X17,
  X18,
  X19,
  UINT8,
  INT8,
};
int32_t PaddleDataTypeSize(int32_t paddle_dtype);

struct TensorInfo {
  std::string name;
  std::vector<int64_t> shape;
  int64_t Rank() const;
  int32_t dtype;
  bool is_tensor_array = false;

  TensorInfo() {}
  TensorInfo(const std::string &_name, const std::vector<int64_t> &_shape,
             const int32_t &_dtype);

  TensorInfo(const TensorInfo &info);
};

struct Weight {
  std::vector<char> buffer;
  std::vector<int32_t> shape;
  int32_t dtype;

  template <typename T>
  void set(int32_t data_type, const std::vector<int64_t> &dims,
           const std::vector<T> &data);

  template <typename T> void get(std::vector<T> *data) const;
};

}  // namespace paddle2onnx
