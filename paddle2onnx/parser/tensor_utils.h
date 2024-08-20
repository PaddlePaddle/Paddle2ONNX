#pragma once
#include <cstdint>
#include <algorithm>
#include <cassert>
#include <numeric>
#include <type_traits>
#include <string>
#include <vector>
#include <cstring>
#include "paddle2onnx/utils/utils.h"

namespace paddle2onnx{
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
  TensorInfo(const std::string& _name, const std::vector<int64_t>& _shape,
             const int32_t& _dtype);

  TensorInfo(const TensorInfo& info);
};

struct Weight {
  std::vector<char> buffer;
  std::vector<int32_t> shape;
  int32_t dtype;

  template <typename T>
  void set(int32_t data_type, const std::vector<int64_t>& dims,
           const std::vector<T>& data);

  template <typename T>
  void get(std::vector<T>* data) const;
};

}
