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
//
// Standalone PIR parser — no dependency on libpaddle.so.
// Uses nlohmann/json for model deserialization.

#pragma once

#include <map>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include "paddle2onnx/pir/pir_program.h"
#include "paddle2onnx/parser/tensor_utils.h"
#include "paddle2onnx/proto/p2o_paddle.pb.h"

namespace paddle2onnx {

class PaddlePirParser {
 public:
  using ScalarData = std::variant<double, float, int64_t, int32_t, bool>;

  bool Init(const std::string& _model, const std::string& _params = "");

  std::map<std::string, Weight> params;
  std::shared_ptr<pir::Program> pir_program_;
  std::vector<TensorInfo> inputs;
  std::vector<TensorInfo> outputs;
  bool is_quantized_model = false;

  // Flat list of ops in the main block
  std::vector<pir::Operation*> global_blocks_ops;
  mutable std::vector<pir::Operation*> sub_blocks_ops;
  std::set<pir::Operation*> total_blocks_ops;

  // While op value mappings (keyed by value ID instead of ValueImpl*)
  mutable std::map<int64_t, int64_t> while_op_values_args_map;
  mutable std::map<int64_t, std::string> while_op_args_name_map;

  explicit PaddlePirParser(bool verbose) : verbose_(verbose) {}

  int NumOfBlocks() const;
  int NumOfProgramOps() const;

  TensorInfo GetTensorInfo(const std::string& name,
                           const pir::Type& value_type) const;
  std::vector<TensorInfo> GetTensorInfo(const pir::Value& value) const;
  std::vector<TensorInfo> GetTensorInfo(const pir::Value& value,
                                        std::string name) const;
  std::vector<TensorInfo> GetSubBlockValueTensorInfo(
      const pir::Value& value) const;

  bool OpIsAttrVar(int64_t op_id,
                   const std::string& name,
                   bool if_in_sub_block) const;
  bool OpHasInput(int64_t op_id,
                  const std::string& input_name,
                  bool if_in_sub_block) const;
  bool OpHasOutput(int64_t op_id,
                   const std::string& output_name,
                   bool if_in_sub_block) const;

  void GetOpAttr(const pir::Operation* op,
                 const std::string& name,
                 int64_t* res) const;
  void GetOpAttr(const pir::Operation* op,
                 const std::string& name,
                 float* res) const;
  void GetOpAttr(const pir::Operation* op,
                 const std::string& name,
                 double* res) const;
  void GetOpAttr(const pir::Operation* op,
                 const std::string& name,
                 bool* res) const;
  void GetOpAttr(const pir::Operation* op,
                 const std::string& name,
                 std::string* res) const;
  void GetOpAttr(const pir::Operation* op,
                 const std::string& name,
                 std::vector<int64_t>* res) const;
  void GetOpAttr(const pir::Operation* op,
                 const std::string& name,
                 std::vector<float>* res) const;
  void GetOpAttr(const pir::Operation* op,
                 const std::string& name,
                 std::vector<double>* res) const;
  void GetOpAttr(const pir::Operation* op,
                 const std::string& name,
                 std::vector<bool>* res) const;
  bool OpHasAttr(const pir::Operation* op, const std::string& name) const;

  void GetOpScalarValue(int64_t op_id,
                        bool if_in_sub_block,
                        const std::string& scalar_attr_name,
                        ScalarData* scalar_data) const;

  std::string GetSubBlockOpOutputName(const pir::Value& source) const;
  std::vector<TensorInfo> GetOpInput(int64_t op_id,
                                     int64_t input_idx,
                                     bool if_in_sub_block) const;
  std::vector<TensorInfo> GetOpOutput(int64_t op_id,
                                      int64_t output_idx,
                                      bool if_in_sub_block) const;
  std::string GetOpArgName(int64_t op_id,
                           std::string name,
                           bool if_in_sub_block) const;
  int32_t GetOpInputOutputName2Idx(int64_t op_id,
                                   std::string name,
                                   bool is_input,
                                   bool if_in_subblock) const;
  void GetSubBlockOpOutputName(
      std::vector<pir::Operation*> block_op_lists) const;

  bool IsConstantTensor(int64_t op_id,
                        int64_t input_idx,
                        bool if_in_sub_block) const;
  std::string GetOpOutputName(const pir::Value& source) const;

  template <typename T>
  bool TryGetTensorValue(int64_t op_id,
                         int64_t input_idx,
                         std::vector<T>* data,
                         bool if_in_sub_block = false) const {
    pir::Operation* temp_op =
        if_in_sub_block ? sub_blocks_ops[op_id] : global_blocks_ops[op_id];
    if (input_idx < 0 || input_idx >= static_cast<int64_t>(temp_op->num_operands()))
      return false;
    auto tensor_infos =
        GetTensorInfo(temp_op->operand(input_idx).source());
    if (tensor_infos.empty()) return false;
    TensorInfo tensor_info = tensor_infos[0];
    auto iter = params.find(tensor_info.name);
    if (iter != params.end()) {
      (iter->second).get(data);
      return true;
    }
    // Walk up the chain to find a constant op (full, fulls, value, values)
    pir::Operation* op = FindDefiningOp(op_id, input_idx, if_in_sub_block);
    if (!op) return false;

    std::string attr_value = "value";
    std::string attr_values = "values";
    if (!op->HasAttribute(attr_value) && !op->HasAttribute(attr_values))
      return false;

    std::string attr_name = op->HasAttribute(attr_value) ? attr_value : attr_values;
    auto attr = op->attribute(attr_name);
    if (!attr.valid()) return false;

    auto vec = attr.AsVector();
    if (vec.empty()) return false;

    data->clear();
    for (auto& elem : vec) {
      data->push_back(static_cast<T>(elem.AsFloat()));
    }
    return true;
  }

  template <typename T>
  bool TryGetTensorValue(int64_t op_id,
                         int64_t input_idx,
                         T* data,
                         bool if_in_sub_block = false) const {
    pir::Operation* temp_op =
        if_in_sub_block ? sub_blocks_ops[op_id] : global_blocks_ops[op_id];
    if (input_idx < 0 || input_idx >= static_cast<int64_t>(temp_op->num_operands()))
      return false;
    auto tensor_infos =
        GetTensorInfo(temp_op->operand(input_idx).source());
    if (tensor_infos.empty()) return false;
    pir::Operation* op = FindDefiningOp(op_id, input_idx, if_in_sub_block);
    if (!op) return false;

    std::string attr_value = "value";
    std::string attr_values = "values";
    if (!op->HasAttribute(attr_value) && !op->HasAttribute(attr_values))
      return false;

    std::string attr_name = op->HasAttribute(attr_value) ? attr_value : attr_values;
    auto attr = op->attribute(attr_name);
    if (!attr.valid()) return false;

    *data = static_cast<T>(attr.AsFloat());
    return true;
  }

  void SetTensorArrayName(int64_t op_id,
                          bool if_in_sub_block,
                          std::string tensor_arr_name) const;
  std::string GetTensorArrayName(int64_t op_id, bool if_in_sub_block) const;
  std::string GenOpInputOutputName(const std::string& name) const;
  void GetWhileInputValuesAndArgsMappings(
      const pir::Operation* while_op) const;

 private:
  bool verbose_;
  bool LoadModel(const std::string& model);
  bool LoadParams(const std::string& path);
  void GetGlobalBlocksOps();
  void GetGlobalBlockOpOutputName();
  void AddOpOutputName(pir::Operation* op,
                       std::string var_name,
                       int64_t output_idx) const;

  // Walk backwards through defining ops to find a constant op
  pir::Operation* FindDefiningOp(int64_t op_id,
                                 int64_t input_idx,
                                 bool if_in_sub_block) const;

  // Lookup value ID from operand
  static int64_t GetOperandValueId(const pir::Operation* op, int64_t idx);

  P2ODataType PirTypeToOldIrDataType(pir::DataType dtype) const;

  mutable std::unordered_map<std::string, int64_t> _name_counter;
  mutable std::unordered_map<pir::Operation*, std::vector<std::string>>
      _op_outputs;
  mutable std::unordered_map<pir::Operation*, std::string>
      _tensor_arr_mappings;
};

inline std::string convert_pir_op_name(const std::string& pir_name) {
  if (pir_name.find("pd_op.") == 0) {
    return pir_name.substr(6);
  }
  return pir_name;
}

}  // namespace paddle2onnx
