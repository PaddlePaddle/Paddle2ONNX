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
#include <algorithm>
#include <cassert>
#include <type_traits>
#include "paddle2onnx/proto/p2o_paddle.pb.h"

namespace paddle2onnx {

enum P2ODataType { BOOL, INT16, INT32, INT64, FP16, FP32, FP64 };
int32_t PaddleDataTypeSize(int32_t paddle_dtype);

struct TensorInfo {
  std::string name;
  std::vector<int64_t> shape;
  int64_t Rank() { return static_cast<int64_t>(shape.size()); }
  int32_t dtype;
};

struct Weight {
  std::vector<char> buffer;
  std::vector<int32_t> shape;
  int32_t dtype;

  template <typename T>
  void set(int32_t data_type, const std::vector<int64_t>& dims,
           const std::vector<T>& data) {
    buffer.clear();
    shape.clear();
    dtype = data_type;
    buffer.resize(data.size() * PaddleDataTypeSize(dtype));
    memcpy(buffer.data(), data.data(), data.size() * PaddleDataTypeSize(dtype));
    for (auto& d : dims) {
      shape.push_back(d);
    }
  }
};

class PaddleParser {
 public:
  // recording variable name:id for each block of a program
  std::vector<std::map<std::string, int32_t>> _blocks_var_name2id;
  // recoring set of operators for each block
  std::vector<std::vector<const paddle2onnx::framework::proto::OpDesc*>>
      _blocks_ops;
  std::shared_ptr<paddle2onnx::framework::proto::ProgramDesc> prog;
  std::map<std::string, Weight> params;
  std::vector<TensorInfo> inputs;
  std::vector<TensorInfo> outputs;

  // Sometimes the model contains no parameters
  // In this case, we only need the model_file
  // If from_memory_buffer is true, means we read the model from memory instead
  // of disk
  bool Init(const std::string& _model, bool from_memory_buffer = false);
  bool Init(const std::string& _model, const std::string& _params,
            bool from_memory_buffer = false);

  int NumOfBlocks() const;
  int NumOfOps(int block_idx) const;
  const framework::proto::OpDesc GetOpDesc(int32_t block_idx,
                                           int32_t op_idx) const;

  bool OpHasInput(int64_t block_id, int64_t op_id,
                  const std::string& name) const;
  bool OpHasOutput(int64_t block_id, int64_t op_id,
                   const std::string& name) const;

  std::vector<TensorInfo> GetOpInput(int64_t block_id, int64_t op_id,
                                     const std::string& name) const;
  std::vector<TensorInfo> GetOpOutput(int64_t block_id, int64_t op_id,
                                      const std::string& name) const;

  std::string GetOpAttrType(const paddle2onnx::framework::proto::OpDesc& op,
                            const std::string& name) const;

  bool OpHasAttr(const paddle2onnx::framework::proto::OpDesc& op,
                 const std::string& name) const;

  void GetOpAttr(const paddle2onnx::framework::proto::OpDesc& op,
                 const std::string& name, int64_t* res) const;
  void GetOpAttr(const paddle2onnx::framework::proto::OpDesc& op,
                 const std::string& name, float* res) const;
  void GetOpAttr(const paddle2onnx::framework::proto::OpDesc& op,
                 const std::string& name, bool* res) const;
  void GetOpAttr(const paddle2onnx::framework::proto::OpDesc& op,
                 const std::string& name, std::string* res) const;
  void GetOpAttr(const paddle2onnx::framework::proto::OpDesc& op,
                 const std::string& name, std::vector<int64_t>* res) const;
  void GetOpAttr(const paddle2onnx::framework::proto::OpDesc& op,
                 const std::string& name, std::vector<float>* res) const;
  void GetOpAttr(const paddle2onnx::framework::proto::OpDesc& op,
                 const std::string& name, std::vector<double>* res) const;
  bool GetValueFromTensor(const int64_t& block_id, const int64_t& op_id) const;
  bool GetValueFromTensor(const int64_t& block_id, const int64_t& op_id,
                          Weight* param) const;

 private:
  // If the model has same output name in difference operators
  // will fail to convert
  bool ExistsDumplicateTensorName() const;
  void GetBlocksVarName2Id();
  void GetBlocksOps();
  TensorInfo GetTensorInfo(
      const std::string& name,
      const paddle2onnx::framework::proto::BlockDesc& block) const;
  void GetGlobalBlockInputOutputInfo();
  bool GetParamNames(std::vector<std::string>* var_names);
  bool LoadProgram(const std::string& model, bool from_memory_buffer);
  bool LoadParams(const std::string& path);
  bool LoadParamsFromMemoryBuffer(const std::string& buffer);
};

}  // namespace paddle2onnx
