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
#include <fstream>
#include <memory>
#include <string>
#include <vector>
#include "paddle2onnx/proto/framework.pb.h"
#include "paddle2onnx/utils/utils.hpp"

namespace paddle2onnx {

enum P2ODataType { BOOL, INT16, INT32, INT64, FP16, FP32, FP64 };

int32_t data_type_size(int32_t dtype) {
  Assert(dtype != FP16, "Float16 is not supported.");
  if (dtype == P2ODataType::BOOL) {
    return sizeof(bool);
  } else if (dtype == P2ODataType::INT16) {
    return sizeof(int16_t);
  } else if (dtype == P2ODataType::INT32) {
    return sizeof(int32_t);
  } else if (dtype == P2ODataType::INT64) {
    return sizeof(int64_t);
  } else if (dtype == P2ODataType::FP32) {
    return sizeof(float);
  } else if (dtype == P2ODataType::FP64) {
    return sizeof(double);
  } else {
    Assert(false, "Unexpected data type");
  }
  return -1;
}

struct Weight {
  std::vector<char> buffer;
  std::vector<int32_t> shape;
  int32_t dtype;

  template <typename T>
  void set(int32_t data_type, const std::vector<int64_t>& dims,
           const std::vector<T>& data) {
    buffer.clear();
    shape.clear();
    buffer.resize(data.size() * data_type_size(dtype));
    memcpy(buffer.data(), data.data(), data.size() * data_type_size(dtype));
    dtype = data_type;
    for (auto& d : dims) {
      shape.push_back(d);
    }
  }
};

bool LoadParams(const std::string& path, std::vector<Weight>* weights) {
  weights->clear();
  std::ifstream is(path, std::ios::in | std::ios::binary);
  if (!is.is_open()) {
    std::cerr << "Cannot open file " << path << " to read." << std::endl;
    return false;
  }
  is.seekg(0, std::ios::end);
  int total_size = is.tellg();
  is.seekg(0, std::ios::beg);

  int read_size = 0;
  while (read_size < total_size) {
    {
      // read version, we don't need this
      uint32_t version;
      read_size += sizeof(version);
      is.read(reinterpret_cast<char*>(&version), sizeof(version));
    }
    {
      // read lod_level, we don't use it
      // this has to be zero, otherwise not support
      uint64_t lod_level;
      read_size += sizeof(lod_level);
      is.read(reinterpret_cast<char*>(&lod_level), sizeof(lod_level));
      Assert(lod_level == 0,
             "Paddle2ONNX only support weight with lod_level = 1.");
    }
    {
      // Another version, we don't use it
      uint32_t version;
      read_size += sizeof(version);
      is.read(reinterpret_cast<char*>(&version), sizeof(version));
    }
    {
      // read size of TensorDesc
      int32_t size;
      read_size += sizeof(size);
      is.read(reinterpret_cast<char*>(&size), sizeof(size));
      // read TensorDesc
      std::unique_ptr<char[]> buf(new char[size]);
      read_size += size;
      is.read(reinterpret_cast<char*>(buf.get()), size);

      std::unique_ptr<paddle2onnx::framework::proto::VarType_TensorDesc>
          tensor_desc(new paddle2onnx::framework::proto::VarType_TensorDesc());
      tensor_desc->ParseFromArray(buf.get(), size);

      Weight weight;

      int32_t numel = 1;
      int32_t data_type = tensor_desc->data_type();
      weight.dtype = data_type;
      for (auto i = 0; i < tensor_desc->dims().size(); ++i) {
        numel *= tensor_desc->dims()[i];
        weight.shape.push_back(tensor_desc->dims()[i]);
      }

      // read weight data
      weight.buffer.resize(numel * data_type_size(data_type));
      read_size += numel * data_type_size(data_type);
      is.read(weight.buffer.data(), numel * data_type_size(data_type));
      weights->push_back(std::move(weight));
    }
  }
  is.close();
  return true;
}

}  // namespace paddle2onnx
