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
#include <numeric>
#include <string>
#include <vector>
#include "deploykit/utils/utils.h"

namespace deploykit {

enum PaddleDataType {
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
  INT8
};
inline int32_t PaddleDataTypeSize(int32_t paddle_dtype) {
  Assert(paddle_dtype != FP16, "Float16 is not supported.");
  if (paddle_dtype == PaddleDataType::BOOL) {
    return sizeof(bool);
  } else if (paddle_dtype == PaddleDataType::INT16) {
    return sizeof(int16_t);
  } else if (paddle_dtype == PaddleDataType::INT32) {
    return sizeof(int32_t);
  } else if (paddle_dtype == PaddleDataType::INT64) {
    return sizeof(int64_t);
  } else if (paddle_dtype == PaddleDataType::FP32) {
    return sizeof(float);
  } else if (paddle_dtype == PaddleDataType::FP64) {
    return sizeof(double);
  } else if (paddle_dtype == PaddleDataType::UINT8) {
    return sizeof(uint8_t);
  } else {
    Assert(false, "Unexpected data type: " + std::to_string(paddle_dtype));
  }
  return -1;
}

struct DataBlob {
  std::vector<char> data;
  std::vector<int64_t> shape;
  std::string name;
  int dtype;

  // Only for Python API, will skip the memory copy from pybind11::array_t
  // When py_array_t is not nullptr, the std::vector<char> data will be ignored
  void* py_array_t = nullptr;

  DataBlob() {}
  explicit DataBlob(const std::string& blob_name) { name = blob_name; }

  void* GetData() {
    if (py_array_t != nullptr) {
      return py_array_t;
    }
    return data.data();
  }

  void Resize(const std::vector<int>& new_shape, const int& data_type) {
    dtype = data_type;
    shape.assign(new_shape.begin(), new_shape.end());
    int unit = PaddleDataTypeSize(data_type);
    int total_size =
        std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
    data.resize(total_size);
  }

  int Nbytes() const { return Numel() * PaddleDataTypeSize(dtype); }

  int Numel() const {
    return std::accumulate(shape.begin(), shape.end(), 1,
                           std::multiplies<int>());
  }
};
}  // namespace deploykit
