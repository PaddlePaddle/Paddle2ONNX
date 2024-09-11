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
#include <map>
#include "paddle/common/errors.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/pir/include/core/operation_utils.h"
#include "paddle/pir/include/core/program.h"
#include "paddle/pir/include/core/value.h"
#include "paddle2onnx/parser/tensor_utils.h"
#include "paddle2onnx/proto/p2o_paddle.pb.h"
namespace paddle2onnx {
class PaddlePirParser {
 public:
  bool Init(const std::string &_model, const std::string &_params = "");
  std::map<std::string, Weight> params;
  std::shared_ptr<pir::Program> pir_program_;
  std::vector<TensorInfo> inputs;
  std::vector<TensorInfo> outputs;
  bool is_quantized_model = false;  // If the Paddle model is a quantized
                                    // model,set is_quantized_model to be true
  // recoring set of operators for global block
  std::vector<pir::Operation *> global_blocks_ops;
  int NumOfBlocks() const;
  // int NumOfOps(int block_idx) const;
  int NumOfProgramOps() const;
  // recoring set of operators for pir global block
  TensorInfo GetTensorInfo(std::string name, const pir::Operation *op);
  void GetOpAttr(const pir::Operation *op,
                 const std::string &name,
                 int64_t *res) const;
  void GetOpAttr(const pir::Operation *op,
                 const std::string &name,
                 float *res) const;
  void GetOpAttr(const pir::Operation *op,
                 const std::string &name,
                 bool *res) const;
  void GetOpAttr(const pir::Operation *op,
                 const std::string &name,
                 std::string *res) const;
  void GetOpAttr(const pir::Operation *op,
                 const std::string &name,
                 std::vector<int64_t> *res) const;
  void GetOpAttr(const pir::Operation *op,
                 const std::string &name,
                 std::vector<float> *res) const;
  void GetOpAttr(const pir::Operation *op,
                 const std::string &name,
                 std::vector<double> *res) const;
  bool OpHasAttr(pir::Operation *op, const std::string &name) const;
  std::vector<TensorInfo> GetOpInput(const pir::Operation *op, const std::string& name, int input_idx);
  std::vector<TensorInfo> GetOpOutput(const pir::Operation *op, const std::string& name, int output_idx);
  std::string GenOpInputOutputName(const std::string& name);

 private:
  bool LoadProgram(const std::string &model);
  bool LoadParams(const std::string &path);
  bool GetParamValueName(std::vector<std::string> *var_names);
  void GetGlobalBlocksOps();
  void GetGlobalBlockInputOutputInfo();
  std::vector<std::map<std::string, int64_t>> _constant_ops;
  std::unordered_map<std::string, int64_t> _name_counter;
  
};
}  // namespace paddle2onnx
