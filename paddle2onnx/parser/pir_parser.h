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

 private:
  bool LoadProgram(const std::string &model);
  bool LoadParams(const std::string &path);
  bool GetParamValueName(std::vector<std::string> *var_names);
  void GetGlobalBlocksOps();
  void GetGlobalBlockInputOutputInfo();
  std::vector<std::map<std::string, int64_t>> _constant_ops;
};

// template<typename T>
// void PaddlePirParser::GetOpAttr(const pir::Operation* op,const std::string&
// name, T* res) const{
//     bool found=false;
//     for (auto &pair : op->attributes()) {
//     if (pair.first == name) {
//         found = true;
//         if(std::is_same<T, int32_t*>::value || std::is_same<T,
//         int64_t*>::value){
//             if(pair.second.isa<pir::Int32Attribute>()){
//                 *res = pair.second.dyn_cast<::pir::Int32Attribute>().data();
//             }else{
//                 *res = pair.second.dyn_cast<::pir::Int64Attribute>().data();
//             }
//         }else if(std::is_same<T, float*>::value){
//             if(pair.second.isa<pir::FloatAttribute>()){
//                 *res = pair.second.dyn_cast<::pir::FloatAttribute>().data();
//             }
//         }else if(std::is_same<T, bool*>::value){
//             if(pair.second.isa<pir::BoolAttribute>()){
//                 *res = pair.second.dyn_cast<::pir::BoolAttribute>().data();
//             }
//         }else if(std::is_same<T, std::string*>::value){
//             if(pair.second.isa<pir::StrAttribute>()){
//                 *res =
//                 pair.second.dyn_cast<::pir::StrAttribute>().AsString();
//             }
//         }else if(std::is_same<T, std::vector<int64_t>*>::value){
//             if(pair.second.isa<pir::ArrayAttribute>()){
//                 auto array_list =
//                 pair.second.dyn_cast<::pir::ArrayAttribute>().AsVector();
//                 if(array_list.size() > 0){
//                     PADDLE_ENFORCE_EQ(array_list[0].isa<::pir::Int64Attribute>(),
//                       true,
//                       ::common::errors::Unimplemented(
//                           "the 0th elementwise MUST be ir::Int64Attribute"));
//                     for (size_t i = 0; i < array_list.size(); ++i) {
//                         res->push_back(array_list[i].dyn_cast<::pir::Int64Attribute>().data());
//                     }
//                 }
//             }
//         }else{
//             PADDLE_THROW(
//                 ::common::errors::InvalidArgument("unsupported attribute
//                 type"));
//         }

//         break;
//     }
//     }
//     Assert(found, "Cannot found attribute " + name + " in op: " +
//     op->name());
// }

}  // namespace paddle2onnx
