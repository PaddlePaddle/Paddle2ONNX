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

#include "paddle2onnx/parser/tensor_utils.h"

namespace paddle2onnx {

int64_t TensorInfo::Rank() const { return static_cast<int64_t>(shape.size()); }
TensorInfo::TensorInfo(const std::string &_name,
                       const std::vector<int64_t> &_shape,
                       const int32_t &_dtype) {
  name = _name;
  shape.assign(_shape.begin(), _shape.end());
  dtype = _dtype;
}
TensorInfo::TensorInfo(const TensorInfo &info) {
  name = info.name;
  shape.assign(info.shape.begin(), info.shape.end());
  dtype = info.dtype;
  is_tensor_array = info.is_tensor_array;
}

// Weight
template <typename T>
void Weight::set(int32_t data_type, const std::vector<int64_t> &dims,
                 const std::vector<T> &data) {
  buffer.clear();
  shape.clear();
  dtype = data_type;
  buffer.resize(data.size() * PaddleDataTypeSize(dtype));
  memcpy(buffer.data(), data.data(), data.size() * PaddleDataTypeSize(dtype));
  for (auto &d : dims) {
    shape.push_back(d);
  }

  template <typename T> void Weight::get(std::vector<T> * data) const {
    int64_t nums = std::accumulate(std::begin(shape), std::end(shape), 1,
                                   std::multiplies<int64_t>());
    data->resize(nums);
    if (dtype == P2ODataType::INT64) {
      std::vector<int64_t> value(nums);
      memcpy(value.data(), buffer.data(), nums * sizeof(int64_t));
      data->assign(value.begin(), value.end());
    } else if (dtype == P2ODataType::INT32) {
      std::vector<int32_t> value(nums);
      memcpy(value.data(), buffer.data(), nums * sizeof(int32_t));
      data->assign(value.begin(), value.end());
    } else if (dtype == P2ODataType::INT8) {
      std::vector<int8_t> value(nums);
      memcpy(value.data(), buffer.data(), nums * sizeof(int8_t));
      data->assign(value.begin(), value.end());
    } else if (dtype == P2ODataType::FP32) {
      std::vector<float> value(nums);
      memcpy(value.data(), buffer.data(), nums * sizeof(float));
      data->assign(value.begin(), value.end());
    } else if (dtype == P2ODataType::FP64) {
      std::vector<double> value(nums);
      memcpy(value.data(), buffer.data(), nums * sizeof(double));
      data->assign(value.begin(), value.end());
    } else {
      Assert(false,
             "Weight::get() only support int64/int32/int8/float32/float64.");
    }
  }
}
int32_t PaddleDataTypeSize(int32_t paddle_dtype) {
  if (paddle_dtype == P2ODataType::BOOL) {
    return sizeof(bool);
  } else if (paddle_dtype == P2ODataType::INT8) {
    return sizeof(int8_t);
  } else if (paddle_dtype == P2ODataType::INT16) {
    return sizeof(int16_t);
  } else if (paddle_dtype == P2ODataType::INT32) {
    return sizeof(int32_t);
  } else if (paddle_dtype == P2ODataType::INT64) {
    return sizeof(int64_t);
  } else if (paddle_dtype == P2ODataType::FP32) {
    return sizeof(float);
  } else if (paddle_dtype == P2ODataType::FP16) {
    return sizeof(int16_t);
  } else if (paddle_dtype == P2ODataType::FP64) {
    return sizeof(double);
  } else if (paddle_dtype == P2ODataType::UINT8) {
    return sizeof(uint8_t);
  } else {
    Assert(false, "Unexpected data type: " + std::to_string(paddle_dtype));
  }
  return -1;
}
}  // namespace paddle2onnx
