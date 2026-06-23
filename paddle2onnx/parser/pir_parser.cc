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

#include "paddle2onnx/parser/pir_parser.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "nlohmann/json.hpp"

namespace paddle2onnx {

using Json = nlohmann::json;

// ===========================================================================
// Data type conversion
// ===========================================================================
P2ODataType PaddlePirParser::PirTypeToOldIrDataType(pir::DataType dtype) const {
  switch (dtype) {
    case pir::DataType::FLOAT32:
      return P2ODataType::FLOAT32;
    case pir::DataType::FLOAT64:
      return P2ODataType::FLOAT64;
    case pir::DataType::FLOAT16:
      return P2ODataType::FLOAT16;
    case pir::DataType::BFLOAT16:
      return P2ODataType::BFLOAT16;
    case pir::DataType::INT32:
      return P2ODataType::INT32;
    case pir::DataType::INT64:
      return P2ODataType::INT64;
    case pir::DataType::INT16:
      return P2ODataType::INT16;
    case pir::DataType::INT8:
      return P2ODataType::INT8;
    case pir::DataType::UINT8:
      return P2ODataType::UINT8;
    case pir::DataType::BOOL:
      return P2ODataType::BOOL;
    case pir::DataType::COMPLEX64:
      return P2ODataType::COMPLEX64;
    case pir::DataType::COMPLEX128:
      return P2ODataType::COMPLEX128;
    default:
      return P2ODataType::UNDEFINED;
  }
}

// ===========================================================================
// Model loading
// ===========================================================================
bool PaddlePirParser::LoadModel(const std::string& model) {
  // Read file
  std::ifstream ifs(model, std::ios::in | std::ios::binary);
  if (!ifs.is_open()) {
    P2OLogger() << "[ERROR] Cannot open model file: " << model << std::endl;
    return false;
  }
  std::stringstream ss;
  ss << ifs.rdbuf();
  std::string content = ss.str();

  // Parse as JSON
  if (!pir_program_) {
    pir_program_ = std::make_shared<pir::Program>();
  }
  if (!pir_program_->LoadFromJson(content)) {
    P2OLogger() << "[ERROR] Failed to parse PIR JSON model" << std::endl;
    return false;
  }

  P2OLogger(true) << "Standalone PIR parser loaded model successfully."
                  << std::endl;
  return true;
}

bool PaddlePirParser::LoadParams(const std::string& path) {
  params.clear();
  std::ifstream is(path, std::ios::in | std::ios::binary);
  if (!is.is_open()) {
    P2OLogger() << "[ERROR] Cannot open params file " << path << std::endl;
    return false;
  }

  is.seekg(0, std::ios::end);
  int64_t total_size = is.tellg();
  is.seekg(0, std::ios::beg);

  // Collect parameter names from the program
  std::vector<std::string> var_names;
  if (pir_program_ && pir_program_->block()) {
    for (auto& op : pir_program_->block()->ops()) {
      if (op.name() == "builtin.parameter" &&
          op.HasAttribute("persistable")) {
        auto attr = op.attribute("persistable");
        if (attr.AsBool()) {
          auto name_attr = op.attribute("parameter_name");
          if (name_attr.valid()) {
            var_names.push_back(name_attr.AsString());
          }
        }
      }
    }
  }

  P2OLogger(true) << "Found " << var_names.size() << " parameters."
                  << std::endl;

  int64_t read_size = 0;
  size_t param_idx = 0;
  while (read_size < total_size && param_idx < var_names.size()) {
    uint32_t version;
    read_size += sizeof(version);
    is.read(reinterpret_cast<char*>(&version), sizeof(version));

    uint64_t lod_level;
    read_size += sizeof(lod_level);
    is.read(reinterpret_cast<char*>(&lod_level), sizeof(lod_level));
    if (lod_level != 0) {
      P2OLogger() << "[ERROR] Only support weight with lod_level = 0."
                  << std::endl;
      return false;
    }

    uint32_t version2;
    read_size += sizeof(version2);
    is.read(reinterpret_cast<char*>(&version2), sizeof(version2));

    int32_t tensor_desc_size;
    read_size += sizeof(tensor_desc_size);
    is.read(reinterpret_cast<char*>(&tensor_desc_size), sizeof(tensor_desc_size));

    std::unique_ptr<char[]> buf(new char[tensor_desc_size]);
    read_size += tensor_desc_size;
    is.read(reinterpret_cast<char*>(buf.get()), tensor_desc_size);

    framework::proto::VarType_TensorDesc tensor_desc;
    tensor_desc.ParseFromArray(buf.get(), tensor_desc_size);

    Weight weight;
    int32_t numel = 1;
    weight.dtype = tensor_desc.data_type();
    for (auto i = 0; i < tensor_desc.dims().size(); ++i) {
      numel *= tensor_desc.dims()[i];
      weight.shape.push_back(tensor_desc.dims()[i]);
    }

    weight.buffer.resize(numel * PaddleDataTypeSize(weight.dtype));
    read_size += numel * PaddleDataTypeSize(weight.dtype);
    is.read(weight.buffer.data(), numel * PaddleDataTypeSize(weight.dtype));

    if (param_idx < var_names.size()) {
      params[var_names[param_idx]] = weight;
    }
    param_idx++;
  }

  is.close();
  return true;
}

bool PaddlePirParser::Init(const std::string& _model,
                           const std::string& _params) {
  if (!LoadModel(_model)) {
    P2OLogger() << "[ERROR] Failed to load " << _model << std::endl;
    return false;
  }

  if (!_params.empty()) {
    if (!LoadParams(_params)) {
      P2OLogger() << "[ERROR] Failed to load parameters." << std::endl;
      return false;
    }
  }

  GetGlobalBlocksOps();
  GetGlobalBlockOpOutputName();
  return true;
}

// ===========================================================================
// Op enumeration
// ===========================================================================
void PaddlePirParser::GetGlobalBlocksOps() {
  global_blocks_ops.clear();
  is_quantized_model = false;
  if (!pir_program_ || !pir_program_->block()) return;

  for (auto& op : pir_program_->block()->ops()) {
    if (op.name() != "builtin.parameter") {
      global_blocks_ops.push_back(&op);
    }
  }
}

void PaddlePirParser::GetGlobalBlockOpOutputName() {
  inputs.clear();
  outputs.clear();

  std::unordered_set<std::string> input_names;
  for (auto op : global_blocks_ops) {
    if (op->name() == "pd_op.data" || op->name() == "pd_op.feed") {
      auto name_attr = op->attribute("name");
      std::string input_name = name_attr.AsString();
      if (op->num_results() > 0) {
        inputs.push_back(
            GetTensorInfo(input_name, op->result(0).type()));
      }
      AddOpOutputName(op, input_name, 0);
      input_names.insert(input_name);
    } else if (op->name() == "pd_op.fetch") {
      auto name_attr = op->attribute("name");
      std::string var_name = name_attr.AsString();
      std::string output_name;
      if (op->num_operands() > 0) {
        pir::Value value = op->operand(0).source();
        std::string def_op_name = value.defining_op()
                                      ? value.defining_op()->name()
                                      : "";
        if (input_names.count(var_name) && def_op_name != "pd_op.data" &&
            def_op_name != "pd_op.feed") {
          output_name = var_name + "." + GenOpInputOutputName(op->name());
        } else {
          output_name = var_name;
        }
        auto output_idx = value.result_index();
        if (op->num_results() > 0) {
          outputs.push_back(
              GetTensorInfo(output_name, op->result(0).type()));
        }
        AddOpOutputName(value.defining_op(), output_name, output_idx);
      }
    } else {
      std::string var_name = GenOpInputOutputName(op->name());
      int num_outputs = op->num_results();
      for (int i = 0; i < num_outputs; ++i) {
        auto tmp_var_name = var_name + "." + std::to_string(i);
        AddOpOutputName(op, tmp_var_name, i);
      }
    }
  }
}

void PaddlePirParser::AddOpOutputName(pir::Operation* op,
                                      std::string var_name,
                                      int64_t output_idx) const {
  if (_op_outputs.count(op) == 0) {
    int num_outputs = op->num_results();
    _op_outputs[op] = std::vector<std::string>(num_outputs, "");
  }
  _op_outputs[op][output_idx] = var_name;
}

std::string PaddlePirParser::GetOpOutputName(
    const pir::Value& source) const {
  auto* defining_op = source.defining_op();
  if (!defining_op) return "";

  auto output_idx = source.result_index();
  if (_op_outputs.count(defining_op) == 0 ||
      _op_outputs.at(defining_op).size() <= static_cast<size_t>(output_idx)) {
    return "";
  }
  return _op_outputs[defining_op][output_idx];
}

std::string PaddlePirParser::GenOpInputOutputName(
    const std::string& name) const {
  std::string new_name = "p2o." + name;
  if (_name_counter.find(new_name) != _name_counter.end()) {
    _name_counter[new_name] += 1;
  } else {
    _name_counter[new_name] = 0;
  }
  new_name += "." + std::to_string(_name_counter[new_name]);
  return new_name;
}

// ===========================================================================
// TensorInfo helpers
// ===========================================================================
TensorInfo PaddlePirParser::GetTensorInfo(
    const std::string& name, const pir::Type& value_type) const {
  TensorInfo info;
  info.name = name;
  info.dtype = PirTypeToOldIrDataType(value_type.dtype());
  info.shape = value_type.dims();
  return info;
}

std::vector<TensorInfo> PaddlePirParser::GetTensorInfo(
    const pir::Value& value) const {
  std::vector<TensorInfo> results;
  std::string name = GetOpOutputName(value);
  results.push_back(GetTensorInfo(name, value.type()));
  return results;
}

std::vector<TensorInfo> PaddlePirParser::GetTensorInfo(
    const pir::Value& value, std::string name) const {
  std::vector<TensorInfo> results;
  results.push_back(GetTensorInfo(name, value.type()));
  return results;
}

std::vector<TensorInfo> PaddlePirParser::GetSubBlockValueTensorInfo(
    const pir::Value& value) const {
  std::vector<TensorInfo> results;
  std::string name = GetSubBlockOpOutputName(value);
  results.push_back(GetTensorInfo(name, value.type()));
  return results;
}

std::string PaddlePirParser::GetSubBlockOpOutputName(
    const pir::Value& source) const {
  auto* op = source.defining_op();
  if (!op) return "";
  auto output_idx = source.result_index();
  if (_op_outputs.count(op) == 0 ||
      _op_outputs.at(op).size() <= static_cast<size_t>(output_idx)) {
    return "";
  }
  return _op_outputs[op][output_idx];
}

// ===========================================================================
// Op attribute access
// ===========================================================================
bool PaddlePirParser::OpHasAttr(const pir::Operation* op,
                                const std::string& name) const {
  return op->HasAttribute(name);
}

void PaddlePirParser::GetOpAttr(const pir::Operation* op,
                                const std::string& name,
                                int64_t* res) const {
  auto attr = op->attribute(name);
  if (attr.valid()) {
    if (attr.isa<pir::Int32Attribute>()) {
      *res = attr.AsInt32();
    } else if (attr.isa<pir::Int64Attribute>()) {
      *res = attr.AsInt64();
    }
  }
}

void PaddlePirParser::GetOpAttr(const pir::Operation* op,
                                const std::string& name,
                                float* res) const {
  auto attr = op->attribute(name);
  if (attr.valid()) {
    *res = attr.AsFloat();
  }
}

void PaddlePirParser::GetOpAttr(const pir::Operation* op,
                                const std::string& name,
                                double* res) const {
  auto attr = op->attribute(name);
  if (attr.valid()) {
    *res = attr.AsDouble();
  }
}

void PaddlePirParser::GetOpAttr(const pir::Operation* op,
                                const std::string& name,
                                bool* res) const {
  auto attr = op->attribute(name);
  if (attr.valid()) {
    *res = attr.AsBool();
  }
}

void PaddlePirParser::GetOpAttr(const pir::Operation* op,
                                const std::string& name,
                                std::string* res) const {
  auto attr = op->attribute(name);
  if (attr.valid()) {
    *res = attr.AsString();
  }
}

void PaddlePirParser::GetOpAttr(const pir::Operation* op,
                                const std::string& name,
                                std::vector<int64_t>* res) const {
  auto attr = op->attribute(name);
  if (attr.valid() && attr.isa<pir::ArrayAttribute>()) {
    *res = attr.AsInt64Array();
  }
}

void PaddlePirParser::GetOpAttr(const pir::Operation* op,
                                const std::string& name,
                                std::vector<float>* res) const {
  auto attr = op->attribute(name);
  if (attr.valid() && attr.isa<pir::ArrayAttribute>()) {
    *res = attr.AsFloatArray();
  }
}

void PaddlePirParser::GetOpAttr(const pir::Operation* op,
                                const std::string& name,
                                std::vector<double>* res) const {
  auto attr = op->attribute(name);
  if (attr.valid() && attr.isa<pir::ArrayAttribute>()) {
    *res = attr.AsDoubleArray();
  }
}

void PaddlePirParser::GetOpAttr(const pir::Operation* op,
                                const std::string& name,
                                std::vector<bool>* res) const {
  auto attr = op->attribute(name);
  if (attr.valid() && attr.isa<pir::ArrayAttribute>()) {
    *res = attr.AsBoolArray();
  }
}

// ===========================================================================
// Input / output access
// ===========================================================================
bool PaddlePirParser::OpHasInput(int64_t op_id,
                                 const std::string& input_name,
                                 bool if_in_sub_block) const {
  // In standalone PIR, inputs are positional. Named lookup is best-effort.
  // Return true if the named input exists in the op's operand list.
  return true;  // default to true for safety; PIR ops always have positional
                // inputs
}

bool PaddlePirParser::OpHasOutput(int64_t op_id,
                                  const std::string& output_name,
                                  bool if_in_sub_block) const {
  return true;  // same reasoning as OpHasInput
}

int32_t PaddlePirParser::GetOpInputOutputName2Idx(
    int64_t op_id,
    std::string name,
    bool is_input,
    bool if_in_subblock) const {
  // Named input/output lookup not needed for PIR ops (they use positional
  // indices). Return -1 to signal "not found by name".
  return -1;
}

std::string PaddlePirParser::GetOpArgName(
    int64_t op_id, std::string name, bool if_in_sub_block) const {
  return name;  // passthrough
}

std::vector<TensorInfo> PaddlePirParser::GetOpInput(
    int64_t op_id, int64_t input_idx, bool if_in_sub_block) const {
  pir::Operation* op =
      if_in_sub_block ? sub_blocks_ops[op_id] : global_blocks_ops[op_id];
  if (input_idx < 0 || input_idx >= static_cast<int64_t>(op->num_operands()))
    return {};
  return GetTensorInfo(op->operand(input_idx).source());
}

std::vector<TensorInfo> PaddlePirParser::GetOpOutput(
    int64_t op_id, int64_t output_idx, bool if_in_sub_block) const {
  pir::Operation* op =
      if_in_sub_block ? sub_blocks_ops[op_id] : global_blocks_ops[op_id];
  if (output_idx < 0 ||
      output_idx >= static_cast<int64_t>(op->num_results()))
    return {};
  return GetTensorInfo(op->result(output_idx));
}

// ===========================================================================
// Other helpers
// ===========================================================================
bool PaddlePirParser::OpIsAttrVar(int64_t op_id,
                                  const std::string& name,
                                  bool if_in_sub_block) const {
  return false;
}

void PaddlePirParser::GetOpScalarValue(
    int64_t op_id,
    bool if_in_sub_block,
    const std::string& scalar_attr_name,
    ScalarData* scalar_data) const {
  pir::Operation* op =
      if_in_sub_block ? sub_blocks_ops[op_id] : global_blocks_ops[op_id];
  auto attr = op->attribute(scalar_attr_name);
  if (!attr.valid()) return;

  if (attr.isa<pir::Int64Attribute>()) {
    *scalar_data = attr.AsInt64();
  } else if (attr.isa<pir::Int32Attribute>()) {
    *scalar_data = attr.AsInt32();
  } else if (attr.isa<pir::FloatAttribute>()) {
    *scalar_data = attr.AsFloat();
  } else if (attr.isa<pir::DoubleAttribute>()) {
    *scalar_data = attr.AsDouble();
  } else if (attr.isa<pir::BoolAttribute>()) {
    *scalar_data = attr.AsBool();
  }
}

bool PaddlePirParser::IsConstantTensor(int64_t op_id,
                                       int64_t input_idx,
                                       bool if_in_sub_block) const {
  pir::Operation* op =
      if_in_sub_block ? sub_blocks_ops[op_id] : global_blocks_ops[op_id];
  if (input_idx < 0 || input_idx >= static_cast<int64_t>(op->num_operands()))
    return false;
  auto value = op->operand(input_idx).source();
  auto* def_op = value.defining_op();
  if (!def_op) return false;

  // Check if the defining op is a constant-like op
  const auto& name = def_op->name();
  return name == "pd_op.full" || name == "pd_op.full_int_array" ||
         name == "pd_op.full_with_tensor" || name.find("constant") != std::string::npos;
}

pir::Operation* PaddlePirParser::FindDefiningOp(
    int64_t op_id, int64_t input_idx, bool if_in_sub_block) const {
  pir::Operation* temp_op =
      if_in_sub_block ? sub_blocks_ops[op_id] : global_blocks_ops[op_id];
  if (input_idx < 0 || input_idx >= static_cast<int64_t>(temp_op->num_operands()))
    return nullptr;
  
  auto value = temp_op->operand(input_idx).source();
  pir::Operation* op = value.defining_op();
  
  // Walk up the chain to find a constant op
  // (full, full_int_array, or has attributes 'value'/'values')
  while (op && op->num_operands() > 0 && !op->HasAttribute("value") &&
         !op->HasAttribute("values")) {
    if (op->num_operands() > 0) {
      auto v = op->operand(0).source();
      op = v.defining_op();
    } else {
      break;
    }
  }
  return op;
}

int64_t PaddlePirParser::GetOperandValueId(const pir::Operation* op,
                                           int64_t idx) {
  if (idx < 0 || idx >= static_cast<int64_t>(op->num_operands())) return -1;
  return op->operand(idx).value_id;
}

int PaddlePirParser::NumOfBlocks() const { return 1; }

int PaddlePirParser::NumOfProgramOps() const {
  if (!pir_program_ || !pir_program_->block()) return 0;
  return pir_program_->block()->ops().size();
}

void PaddlePirParser::GetSubBlockOpOutputName(
    std::vector<pir::Operation*> block_op_lists) const {
  for (auto op : block_op_lists) {
    std::string new_name = "p2o.sub_block." + op->name();
    if (_name_counter.find(new_name) != _name_counter.end()) {
      _name_counter[new_name] += 1;
    } else {
      _name_counter[new_name] = 0;
    }
    new_name += "." + std::to_string(_name_counter[new_name]);
    int num_outputs = op->num_results();
    for (int i = 0; i < num_outputs; ++i) {
      std::string var_name = new_name + "." + std::to_string(i);
      if (_op_outputs.count(op) == 0) {
        _op_outputs[op] = std::vector<std::string>(num_outputs, "");
      }
      _op_outputs[op][i] = var_name;
    }
  }
}

void PaddlePirParser::SetTensorArrayName(
    int64_t op_id,
    bool if_in_sub_block,
    std::string tensor_arr_name) const {
  pir::Operation* op =
      if_in_sub_block ? sub_blocks_ops[op_id] : global_blocks_ops[op_id];
  _tensor_arr_mappings[op] = tensor_arr_name;
}

std::string PaddlePirParser::GetTensorArrayName(
    int64_t op_id, bool if_in_sub_block) const {
  pir::Operation* op =
      if_in_sub_block ? sub_blocks_ops[op_id] : global_blocks_ops[op_id];
  auto it = _tensor_arr_mappings.find(op);
  if (it != _tensor_arr_mappings.end()) return it->second;
  return "";
}

}  // namespace paddle2onnx
