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
#include <variant>

#include "paddle/common/errors.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/pir/include/core/operation_utils.h"
#include "paddle/pir/include/core/program.h"
#include "paddle/pir/include/core/value.h"
#include "paddle2onnx/parser/tensor_utils.h"
#include "paddle2onnx/proto/p2o_paddle.pb.h"
#include "paddle/fluid/pir/dialect/operator/ir/control_flow_op.h"
namespace paddle2onnx {
class PaddlePirParser {
 public:
  typedef std::variant<double, float, int64_t, int32_t, bool> ScalarData;
  bool Init(const std::string& _model, const std::string& _params = "");
  std::map<std::string, Weight> params;
  std::shared_ptr<pir::Program> pir_program_;
  std::vector<TensorInfo> inputs;
  std::vector<TensorInfo> outputs;
  bool is_quantized_model = false;  // If the Paddle model is a quantized
                                    // model,set is_quantized_model to be true
  // recoring set of operators for global block
  std::vector<pir::Operation*> global_blocks_ops;
  // recoring set of operators for sub block
  mutable std::vector<pir::Operation*>
      sub_blocks_ops;  // todo(wangmingkai02): delete sub_blocks_ops
  // recoring set of operators for all blocks
  std::set<pir::Operation*> total_blocks_ops;
  // recording args of while op body name info
  mutable std::unordered_map<pir::detail::ValueImpl*, pir::detail::ValueImpl*>
      while_op_values_args_map;
  mutable std::unordered_map<pir::detail::ValueImpl*, std::string>
      while_op_args_name_map;
  int NumOfBlocks() const;
  // int NumOfOps(int block_idx) const;
  int NumOfProgramOps() const;
  // recoring set of operators for pir global block
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
  bool OpHasAttr(pir::Operation* op, const std::string& name) const;

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
  void GetAllSubBlockOpOutputName(
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
    PADDLE_ENFORCE_GT(
        input_idx,
        -1,
        common::errors::InvalidArgument(
            "input_idx should be greater than -1 in TryGetTensorValue."));
    pir::Operation* temp_op =
        if_in_sub_block ? sub_blocks_ops[op_id] : global_blocks_ops[op_id];
    TensorInfo tensor_info =
        GetTensorInfo(temp_op->operand(input_idx).source())[0];
    auto iter = params.find(tensor_info.name);
    if (iter != params.end()) {
      (iter->second).get(data);
      return true;
    }
    // TODO(qzylalala): Need double-check
    std::string attr_name;
    std::string attr_value = "value";
    std::string attr_values = "values";
    pir::Operation* op = temp_op->operand(input_idx).source().defining_op();
    while (op->num_operands() > 0 && !op->HasAttribute(attr_value) &&
           !op->HasAttribute(attr_values)) {
      op = op->operand(0).source().defining_op();
    }
    if (op->HasAttribute(attr_value)) {
      attr_name = attr_value;
    } else if (op->HasAttribute(attr_values)) {
      attr_name = attr_values;
    } else {
      return false;
    }
    int32_t dtype = tensor_info.dtype;
    // PADDLE_ENFORCE_EQ(
    //     op->HasAttribute(attr_name),
    //     true,
    //     common::errors::InvalidArgument(
    //         "Cannot found attribute '%s' in op %s", attr_name, op->name()));

    auto array_list =
        op->attribute(attr_name).dyn_cast<::pir::ArrayAttribute>().AsVector();
    if (array_list.size() > 0) {
      if (array_list[0].isa<::pir::FloatAttribute>()) {
        std::vector<float> res;
        for (size_t i = 0; i < array_list.size(); ++i) {
          res.push_back(array_list[i].dyn_cast<::pir::FloatAttribute>().data());
        }
        data->assign(res.begin(), res.end());
      } else if (array_list[0].isa<::pir::DoubleAttribute>()) {
        std::vector<double> res;
        for (size_t i = 0; i < array_list.size(); ++i) {
          res.push_back(
              array_list[i].dyn_cast<::pir::DoubleAttribute>().data());
        }
        data->assign(res.begin(), res.end());
      } else if (array_list[0].isa<::pir::Int32Attribute>()) {
        std::vector<int32_t> res;
        for (size_t i = 0; i < array_list.size(); ++i) {
          res.push_back(
              array_list[i].dyn_cast<::pir::Int32Attribute>().data());
        }
        data->assign(res.begin(), res.end());
      } else if (array_list[0].isa<::pir::Int64Attribute>()) {
        std::vector<int64_t> res;
        for (size_t i = 0; i < array_list.size(); ++i) {
          res.push_back(array_list[i].dyn_cast<::pir::Int64Attribute>().data());
        }
        data->assign(res.begin(), res.end());
      } else {
        Assert(false,
               "Only support int32/int64/float32/float64 data type now.");
      }
    } else {
      return false;
    }
    return true;
  }

  template <typename T>
  bool TryGetTensorValue(int64_t op_id,
                         int64_t input_idx,
                         T* data,
                         bool if_in_sub_block = false) const {
    PADDLE_ENFORCE_GT(
        input_idx,
        -1,
        common::errors::InvalidArgument(
            "input_idx should be greater than -1 in TryGetTensorValue."));
    pir::Operation* temp_op =
        if_in_sub_block ? sub_blocks_ops[op_id] : global_blocks_ops[op_id];
    TensorInfo tensor_info =
        GetTensorInfo(temp_op->operand(input_idx).source())[0];
    pir::Operation* op = temp_op->operand(input_idx).source().defining_op();
    // PADDLE_ENFORCE_EQ(
    //   op->HasAttribute(attr_name),
    //   true,
    //   common::errors::InvalidArgument(
    //     "Cannot found attribute '%s' in op %s", attr_name, op->name()));
    // TODO(qzylalala): Need double-check
    std::string attr_name;
    std::string attr_value = "value";
    std::string attr_values = "values";
    while (op->num_operands() > 0 && !op->HasAttribute(attr_value) &&
           !op->HasAttribute(attr_values)) {
      op = op->operand(0).source().defining_op();
    }
    if (op->HasAttribute(attr_value)) {
      attr_name = attr_value;
    } else if (op->HasAttribute(attr_values)) {
      attr_name = attr_values;
    } else {
      return false;
    }
    auto value = op->attribute(attr_name);
    if (value.isa<pir::Int32Attribute>()) {
      *data = value.dyn_cast<::pir::Int32Attribute>().data();
    } else if (value.isa<pir::Int64Attribute>()) {
      *data = value.dyn_cast<::pir::Int64Attribute>().data();
    } else if (value.isa<pir::FloatAttribute>()) {
      *data = value.dyn_cast<::pir::FloatAttribute>().data();
    } else if (value.isa<pir::DoubleAttribute>()) {
      *data = value.dyn_cast<::pir::DoubleAttribute>().data();
    } else {
      if (value.isa<pir::ArrayAttribute>()) {
        auto array_list = value.dyn_cast<::pir::ArrayAttribute>().AsVector();
        if (array_list.size() > 0) {
          if (array_list[0].isa<::pir::FloatAttribute>()) {
            *data = array_list[0].dyn_cast<::pir::FloatAttribute>().data();
          } else if (array_list[0].isa<::pir::DoubleAttribute>()) {
            *data = array_list[0].dyn_cast<::pir::DoubleAttribute>().data();
          } else if (array_list[0].isa<::pir::Int32Attribute>()) {
            *data = array_list[0].dyn_cast<::pir::Int32Attribute>().data();
          } else if (array_list[0].isa<::pir::Int64Attribute>()) {
            *data = array_list[0].dyn_cast<::pir::Int64Attribute>().data();
          }
        } else {
          return false;
        }
      } else {
        Assert(false,
               "Only support int32/int64/float32/float64 data type now.");
      }
    }
    return true;
  }

  void SetTensorArrayName(int64_t op_id,
                          bool if_in_sub_block,
                          std::string tensor_arr_name) const;
  std::string GetTensorArrayName(int64_t op_id, bool if_in_sub_block) const;
  std::string GenOpInputOutputName(const std::string& name) const;
  void GetWhileInputValuesAndArgsMappings(
      paddle::dialect::WhileOp *while_op) const;

 private:
  bool IsAttrVar(const pir::Operation* op, const int64_t& attr_id) const;
  bool LoadProgram(const std::string& model);
  bool LoadParams(const std::string& path);
  bool GetParamValueName(std::vector<std::string>* var_names);
  void GetGlobalBlocksOps();
  void GetAllBlocksOpsSet(pir::Block *block);
  void GetGlobalBlockInputOutputInfo();
  void GetGlobalBlockInputValueName();
  void GetGlobalBlockOutputValueName();
  void GetAllOpOutputName();
  void AddOpOutputName(pir::Operation* op,
                       std::string var_name,
                       int64_t output_idx) const;

  void GetOpArgNameMappings();
  P2ODataType TransPirDataType2OldIrDataType(phi::DataType dtype) const;
  mutable std::unordered_map<std::string, int64_t> _name_counter;
  mutable std::unordered_map<pir::Operation*, std::vector<std::string>>
      _op_outputs;
  std::unordered_map<std::string, std::unordered_map<std::string, std::string>>
      _op_arg_name_mappings;
  mutable std::unordered_map<pir::Operation*, std::string> _tensor_arr_mappings;
};
}  // namespace paddle2onnx
