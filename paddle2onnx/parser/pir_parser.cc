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
#include "paddle2onnx/parser/pir_parser.h"

#include <algorithm>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "paddle/common/ddim.h"
#include "paddle/fluid/ir_adaptor/translator/op_compat_info.h"
#include "paddle/fluid/pir/dialect/operator/ir/control_flow_op.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_attribute.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_dialect.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_type.h"
#include "paddle/fluid/pir/dialect/operator/utils/op_yaml_info_parser.h"
#include "paddle/fluid/pir/dialect/operator/utils/utils.h"
#include "paddle/fluid/pir/serialize_deserialize/include/interface.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/pir/include/core/builtin_dialect.h"
#include "paddle/pir/include/core/builtin_op.h"
#include "paddle/pir/include/core/ir_context.h"
#include "paddle2onnx/mapper/data_helper.h"
#include "paddle2onnx/proto/p2o_paddle.pb.h"

phi::DataType TransToPhiDataType(pir::Type dtype) {
  if (dtype.isa<pir::BFloat16Type>()) {
    return phi::DataType::BFLOAT16;
  } else if (dtype.isa<pir::Float16Type>()) {
    return phi::DataType::FLOAT16;
  } else if (dtype.isa<pir::Float32Type>()) {
    return phi::DataType::FLOAT32;
  } else if (dtype.isa<pir::Float64Type>()) {
    return phi::DataType::FLOAT64;
  } else if (dtype.isa<pir::UInt8Type>()) {
    return phi::DataType::UINT8;
  } else if (dtype.isa<pir::Int8Type>()) {
    return phi::DataType::INT8;
  } else if (dtype.isa<pir::Int16Type>()) {
    return phi::DataType::INT16;
  } else if (dtype.isa<pir::Int32Type>()) {
    return phi::DataType::INT32;
  } else if (dtype.isa<pir::Int64Type>()) {
    return phi::DataType::INT64;
  } else if (dtype.isa<pir::IndexType>()) {
    return phi::DataType::INT32;
  } else if (dtype.isa<pir::BoolType>()) {
    return phi::DataType::BOOL;
  } else if (dtype.isa<pir::Complex64Type>()) {
    return phi::DataType::COMPLEX64;
  } else if (dtype.isa<pir::Complex128Type>()) {
    return phi::DataType::COMPLEX128;
  } else if (dtype.isa<pir::Float8E4M3FNType>()) {
    return phi::DataType::FLOAT8_E4M3FN;
  } else if (dtype.isa<pir::Float8E5M2Type>()) {
    return phi::DataType::FLOAT8_E5M2;
  } else {
    std::cerr << "Unsupported data type: " << dtype << std::endl;
    return phi::DataType::UNDEFINED;
  }
}

namespace paddle2onnx {
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
void PaddlePirParser::AddOpOutputName(pir::Operation* op,
                                      std::string var_name,
                                      int64_t output_idx) const {
  if (_op_outputs.count(op) == 0) {
    int num_outputs = op->num_results();
    _op_outputs[op] = std::vector<std::string>(num_outputs, "");
  }
  _op_outputs[op][output_idx] = var_name;
}

std::string PaddlePirParser::GetOpOutputName(const pir::Value& source) const {
  auto op = source.defining_op();
  auto output_idx = source.dyn_cast<pir::OpResult>().index();
  if (_op_outputs.count(op) == 0 || _op_outputs.at(op).size() <= output_idx) {
    P2OLogger() << "input is a parameter" << std::endl;
    return op->result(0).defining_op<pir::ParameterOp>().param_name();
    // std::cerr << "Can not find output name" << std::endl;
  }
  return _op_outputs[op][output_idx];
}

std::string PaddlePirParser::GetSubBlockOpOutputName(
    const pir::Value& source) const {
  auto name_iter = while_op_args_name_map.find(&(*(source.impl())));
  if (name_iter != while_op_args_name_map.end()) {
    return name_iter->second;
  }
  auto value_iter = while_op_values_args_map.find(&(*(source.impl())));
  pir::Operation* op;
  uint32_t output_idx;
  if (value_iter != while_op_values_args_map.end()) {
    pir::Value value(value_iter->second);
    op = value.defining_op();
    output_idx = value.dyn_cast<pir::OpResult>().index();
  } else {
    op = source.defining_op();
    output_idx = source.dyn_cast<pir::OpResult>().index();
  }
  if (_op_outputs.count(op) == 0 || _op_outputs.at(op).size() <= output_idx) {
    P2OLogger() << "input is a parameter" << std::endl;
    return op->result(0).defining_op<pir::ParameterOp>().param_name();
  }
  return _op_outputs[op][output_idx];
}

void PaddlePirParser::GetAllSubBlockOpOutputName(
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
void PaddlePirParser::GetAllOpOutputName() {
  inputs.clear();
  outputs.clear();
  for (auto op : global_blocks_ops) {
    if (op->name() == "pd_op.data" || op->name() == "pd_op.feed") {
      std::string input_name =
          op->attribute<pir::StrAttribute>("name").AsString();
      std::string var_name =
          input_name + "." + GenOpInputOutputName(op->name());
      inputs.push_back(GetTensorInfo(var_name, op->result(0).type()));
      AddOpOutputName(op, var_name, 0);
    } else if (op->name() == "pd_op.fetch") {
      auto value = op->operand(0).source();
      auto output_idx = value.dyn_cast<pir::OpResult>().index();
      std::string var_name = _op_outputs[value.defining_op()][output_idx];
      outputs.push_back(GetTensorInfo(var_name, op->result(0).type()));
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

void PaddlePirParser::GetOpArgNameMappings() {
  const std::string pir_op_name_prefix = "pd_op.";
  auto& normalizer = paddle::translator::OpNameNormalizer::instance();
  const auto& op_name_mappings = normalizer.GetOpNameMappings();
  const auto& op_arg_name_mappings = normalizer.GetOpArgNameMappings();
  const auto& op_mutable_attribute_infos =
      normalizer.GetOpMutableAttributeInfos();
  for (auto& item : op_arg_name_mappings) {
    std::string op_name =
        pir_op_name_prefix + (op_name_mappings.count(item.first)
                                  ? op_name_mappings.at(item.first)
                                  : item.first);
    std::unordered_map<std::string, std::string> arg_name_mapping;
    for (auto& arg : item.second) {
      arg_name_mapping[arg.second] = arg.first;
    }
    _op_arg_name_mappings[op_name] = arg_name_mapping;
  }

  // mutable attibute name mappings
  for (auto& item : op_mutable_attribute_infos) {
    std::string op_name =
        pir_op_name_prefix + (op_name_mappings.count(item.first)
                                  ? op_name_mappings.at(item.first)
                                  : item.first);
    for (auto& attr : item.second) {
      for (auto& attr_item : attr.second) {
        _op_arg_name_mappings[op_name][attr_item] = attr.first;
      }
    }
  }
}

void PaddlePirParser::GetAllBlocksOpsSet(pir::Block* block) {
  for (auto& op : block->ops()) {
    std::string op_name = op->name();
    if (op_name != "builtin.parameter") {
      total_blocks_ops.insert(op);
      if (op_name == "pd_op.if") {
        auto if_op = op->dyn_cast<paddle::dialect::IfOp>();
        pir::Block& true_block = if_op.true_block();
        GetAllBlocksOpsSet(&true_block);
        pir::Block& false_block = if_op.false_block();
        GetAllBlocksOpsSet(&false_block);
      } else if (op_name == "pd_op.while") {
        auto while_op = op->dyn_cast<paddle::dialect::WhileOp>();
        GetAllBlocksOpsSet(&while_op.body());
      }
    }
  }
}

std::string PaddlePirParser::GetOpArgName(int64_t op_id,
                                          std::string name,
                                          bool if_in_sub_block) const {
  pir::Operation* op;
  if (if_in_sub_block) {
    op = sub_blocks_ops[op_id];
  } else {
    op = global_blocks_ops[op_id];
  }

  pir::IrContext* ctx = pir::IrContext::Instance();
  std::string op_name = op->name();
  if (op->attributes().count("op_name")) {
    op_name =
        op->attributes().at("op_name").dyn_cast<pir::StrAttribute>().AsString();
  }
  if (_op_arg_name_mappings.count(op_name)) {
    name = _op_arg_name_mappings.at(op_name).count(name)
               ? _op_arg_name_mappings.at(op_name).at(name)
               : name;
  } else {
    if (op_name[op_name.size() - 1] == '_') {
      std::string temp_op_name = op_name.substr(0, op_name.size() - 1);
      if (_op_arg_name_mappings.count(temp_op_name)) {
        name = _op_arg_name_mappings.at(temp_op_name).count(name)
                   ? _op_arg_name_mappings.at(temp_op_name).at(name)
                   : name;
      }
    }
  }
  return name;
}

int32_t PaddlePirParser::GetOpInputOutputName2Idx(int64_t op_id,
                                                  std::string name,
                                                  bool is_input,
                                                  bool if_in_subblock) const {
  pir::Operation* op;
  if (if_in_subblock) {
    op = sub_blocks_ops[op_id];
  } else {
    op = global_blocks_ops[op_id];
  }
  pir::IrContext* ctx = pir::IrContext::Instance();
  std::string op_name = op->name();
  if (op->attributes().count("op_name")) {
    op_name =
        op->attributes().at("op_name").dyn_cast<pir::StrAttribute>().AsString();
  }
  pir::OpInfo op_info = ctx->GetRegisteredOpInfo(op_name);
  paddle::dialect::OpYamlInfoParser yaml_parser(
      op_info.GetInterfaceImpl<paddle::dialect::OpYamlInfoInterface>()
          ->get_op_info_(op_name),
      // paddle::dialect::IsLegacyOp(op_name));
      false);
  name = GetOpArgName(op_id, name, if_in_subblock);
  bool exist = is_input ? yaml_parser.InputName2Id().count(name)
                        : yaml_parser.OutputName2Id().count(name);
  if (!exist) {
    P2OLogger() << "Cannot find input/output name '" << name
                << "' in op yaml info of " << op_name << std::endl;
    return -1;
  }
  // PADDLE_ENFORCE_EQ(
  //     exist,
  //     true,
  //     common::errors::InvalidArgument(
  //         "Cannot find input/output name '%s' in op yaml info of %s.",
  //         name, op_name));
  return is_input ? yaml_parser.InputName2Id().at(name)
                  : yaml_parser.OutputName2Id().at(name);
}

bool PaddlePirParser::LoadProgram(const std::string& model) {
  pir::IrContext* ctx = pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();
  ctx->GetOrRegisterDialect<pir::BuiltinDialect>();
  pir_program_ = std::make_shared<pir::Program>(ctx);
  if (!pir::ReadModule(model, pir_program_.get(), /*pir_version*/ 1)) {
    P2OLogger() << "Failed to deserialize PaddlePaddle model." << std::endl;
    return false;
  }
  std::ostringstream print_stream;
  pir_program_.get()->Print(print_stream);
  P2OLogger() << "PIR Program: \n" << print_stream.str() << std::endl;
  return true;
}
bool PaddlePirParser::GetParamValueName(std::vector<std::string>* var_names) {
  var_names->clear();
  P2OLogger() << "Start getting paramas value name from pir::program"
              << std::endl;
  auto global_block = pir_program_->block();
  std::vector<pir::Value> value_list;
  for (auto& op : global_block->ops()) {
    if (op->name() == "builtin.parameter" &&
        op->HasAttribute(kAttrIsPersistable)) {
      auto attrs = op->attribute(kAttrIsPersistable)
                       .dyn_cast<pir::ArrayAttribute>()
                       .AsVector();
      for (uint32_t i = 0; i < attrs.size(); i++) {
        bool is_persistable = attrs[i].dyn_cast<pir::BoolAttribute>().data();
        if (is_persistable) {
          auto value = static_cast<pir::Value>(op->result(i));
          if (auto param_op = value.defining_op<::pir::ParameterOp>()) {
            var_names->push_back(param_op.param_name());
          }
        }
      }
    }
  }

  std::sort(var_names->begin(), var_names->end());
  return true;
}

bool PaddlePirParser::LoadParams(const std::string& path) {
  params.clear();
  std::ifstream is(path, std::ios::in | std::ios::binary);
  if (!is.is_open()) {
    P2OLogger() << "Cannot open file " << path << " to read." << std::endl;
    return false;
  }
  is.seekg(0, std::ios::end);
  int64_t total_size = is.tellg();
  is.seekg(0, std::ios::beg);
  std::vector<std::string> var_names;
  GetParamValueName(&var_names);
  P2OLogger() << "Getting parama's attribute 'param_namefrom' from "
                 "pir::program successfully"
              << std::endl;

  int64_t read_size = 0;
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
             "Paddle2ONNX: Only support weight with lod_level = 0.");
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
      weight.buffer.resize(numel * PaddleDataTypeSize(data_type));
      read_size += numel * PaddleDataTypeSize(data_type);
      is.read(weight.buffer.data(), numel * PaddleDataTypeSize(data_type));
      auto index = params.size();
      if (index >= var_names.size()) {
        P2OLogger() << "Unexcepted situation happend while reading parameters "
                       "of PaddlePaddle pir model."
                    << std::endl;
        return false;
      }
      params[var_names[index]] = weight;
    }
  }
  is.close();
  return true;
}

bool PaddlePirParser::Init(const std::string& _model,
                           const std::string& _params) {
  std::vector<Weight> weights;
  if (!LoadProgram(_model)) {
    P2OLogger() << "Failed to load " << _model << std::endl;
    return false;
  }
  P2OLogger() << "Load PaddlePaddle pir model " << _model << "successfully"
              << std::endl;
  if (_params != "") {
    if (!LoadParams(_params)) {
      P2OLogger() << "Failed to load parameters of PaddlePaddle model"
                  << std::endl;
      return false;
    }
    P2OLogger() << "Load parameters " << _params << " successfully"
                << std::endl;
  }

  // InitBlock();
  GetGlobalBlocksOps();
  GetAllOpOutputName();
  GetOpArgNameMappings();
  GetAllBlocksOpsSet(pir_program_->block());
  return true;
}
int PaddlePirParser::NumOfBlocks() const {
  size_t num_blocks = 0;
  auto top_level_op = pir_program_->module_op();
  for (size_t i = 0; i < top_level_op->num_regions(); ++i) {
    auto& region = top_level_op->region(i);
    num_blocks += region.size();
  }
  return num_blocks;
}

int PaddlePirParser::NumOfProgramOps() const { return pir_program_->num_ops(); }

void PaddlePirParser::GetGlobalBlocksOps() {
  is_quantized_model = false;
  global_blocks_ops.clear();
  auto global_block = pir_program_->block();
  for (auto& op : global_block->ops()) {
    if (op->name() != "builtin.parameter") {
      global_blocks_ops.push_back(op);
    }
  }
}

TensorInfo PaddlePirParser::GetTensorInfo(const std::string& name,
                                          const pir::Type& value_type) const {
  TensorInfo info;
  if (value_type.isa<pir::DenseTensorType>()) {
    // get info.name
    info.name = name;
    // get info.dtype
    auto type = value_type.cast<pir::DenseTensorType>().dtype();
    auto data_type = TransToPhiDataType(type);
    info.dtype = TransPirDataType2OldIrDataType(data_type);
    // get info.shape
    std::vector<int64_t> dims =
        common::vectorize(value_type.cast<pir::DenseTensorType>().dims());
    info.shape = dims;
  } else if (value_type.isa<paddle::dialect::DenseTensorArrayType>()) {
    info.name = name;
    info.is_tensor_array = true;
    auto type =
        value_type.cast<paddle::dialect::DenseTensorArrayType>().dtype();
    auto data_type = TransToPhiDataType(type);
    info.dtype = TransPirDataType2OldIrDataType(data_type);
  } else {
    std::cerr << "only support dense tensor type" << std::endl;
  }
  return info;
}

std::vector<TensorInfo> PaddlePirParser::GetTensorInfo(
    const pir::Value& value) const {
  std::vector<TensorInfo> results;
  if (value.type().isa<pir::VectorType>()) {
    auto vec_type = value.type().cast<pir::VectorType>();
    std::string prefix = GetOpOutputName(value);
    for (int32_t idx = 0; idx < vec_type.size(); idx++) {
      std::string name = prefix + "_" + std::to_string(idx);
      results.push_back(GetTensorInfo(name, vec_type[idx]));
    }
  } else {
    std::string name = GetOpOutputName(value);
    results.push_back(GetTensorInfo(name, value.type()));
  }
  return results;
}

std::vector<TensorInfo> PaddlePirParser::GetTensorInfo(const pir::Value& value,
                                                       std::string name) const {
  std::vector<TensorInfo> results;
  if (value.type().isa<pir::VectorType>()) {
    auto vec_type = value.type().cast<pir::VectorType>();
    std::string prefix = GetOpOutputName(value);
    for (int32_t idx = 0; idx < vec_type.size(); idx++) {
      results.push_back(GetTensorInfo(name, vec_type[idx]));
    }
  } else {
    results.push_back(GetTensorInfo(name, value.type()));
  }
  return results;
}

std::vector<TensorInfo> PaddlePirParser::GetSubBlockValueTensorInfo(
    const pir::Value& value) const {
  std::vector<TensorInfo> results;
  if (value.type().isa<pir::VectorType>()) {
    auto vec_type = value.type().cast<pir::VectorType>();
    std::string prefix = GetSubBlockOpOutputName(value);
    for (int32_t idx = 0; idx < vec_type.size(); idx++) {
      std::string name = prefix + "_" + std::to_string(idx);
      results.push_back(GetTensorInfo(name, vec_type[idx]));
    }
  } else {
    std::string name = GetSubBlockOpOutputName(value);
    results.push_back(GetTensorInfo(name, value.type()));
  }
  return results;
}

bool PaddlePirParser::IsAttrVar(const pir::Operation* op,
                                const int64_t& attr_id) const {
  // TODO(qzylalala): For Resnet50, this interface always return false.
  return false;
}

bool PaddlePirParser::OpIsAttrVar(int64_t op_id,
                                  const std::string& name,
                                  bool if_in_sub_block) const {
  bool is_attr_var = false;
  pir::Operation* op;
  if (if_in_sub_block) {
    op = sub_blocks_ops[op_id];
  } else {
    op = global_blocks_ops[op_id];
  }

  int32_t i = 0;
  for (auto [key, value] : op->attributes()) {
    if (key == name && IsAttrVar(op, i)) {
      is_attr_var = true;
      break;
    }
    i++;
  }

  return is_attr_var;
}

bool PaddlePirParser::OpHasInput(int64_t op_id,
                                 const std::string& input_name,
                                 bool if_in_sub_block) const {
  pir::Operation* op;
  if (if_in_sub_block) {
    op = sub_blocks_ops[op_id];
  } else {
    op = global_blocks_ops[op_id];
  }

  int64_t input_idx =
      GetOpInputOutputName2Idx(op_id, input_name, true, if_in_sub_block);
  return input_idx != -1 && input_idx < op->num_operands() &&
         op->operand(input_idx);
}

bool PaddlePirParser::OpHasOutput(int64_t op_id,
                                  const std::string& output_name,
                                  bool if_in_sub_block) const {
  pir::Operation* op;
  if (if_in_sub_block) {
    op = sub_blocks_ops[op_id];
  } else {
    op = global_blocks_ops[op_id];
  }
  int64_t output_idx =
      GetOpInputOutputName2Idx(op_id, output_name, false, if_in_sub_block);
  return output_idx != -1 && output_idx < op->num_results() &&
         op->result(output_idx);
}

bool PaddlePirParser::OpHasAttr(pir::Operation* op,
                                const std::string& name) const {
  return op->HasAttribute(name);
}

void PaddlePirParser::GetOpAttr(const pir::Operation* op,
                                const std::string& name,
                                int64_t* res) const {
  bool found = false;
  for (auto& pair : op->attributes()) {
    if (pair.first == name) {
      found = true;
      if (pair.second.isa<pir::Int32Attribute>()) {
        *res = pair.second.dyn_cast<::pir::Int32Attribute>().data();
      } else if (pair.second.isa<pir::Int64Attribute>()) {
        *res = pair.second.dyn_cast<::pir::Int64Attribute>().data();
      } else if (pair.second.isa<paddle::dialect::DataTypeAttribute>()) {
        phi::DataType data_type =
            pair.second.dyn_cast<paddle::dialect::DataTypeAttribute>().data();
        *res = TransPirDataType2OldIrDataType(data_type);
      }
      break;
    }
  }
  PADDLE_ENFORCE_EQ(
      found,
      true,
      common::errors::InvalidArgument(
          "Cannot found attribute %s in op %s", name, op->name()));
}

void PaddlePirParser::GetOpAttr(const pir::Operation* op,
                                const std::string& name,
                                float* res) const {
  bool found = false;
  for (auto& pair : op->attributes()) {
    if (pair.first == name) {
      found = true;
      if (pair.second.isa<pir::FloatAttribute>()) {
        *res = pair.second.dyn_cast<::pir::FloatAttribute>().data();
        break;
      }
    }
  }
  PADDLE_ENFORCE_EQ(
      found,
      true,
      common::errors::InvalidArgument(
          "Cannot found attribute %s in op %s", name, op->name()));
}

void PaddlePirParser::GetOpAttr(const pir::Operation* op,
                                const std::string& name,
                                double* res) const {
  bool found = false;
  for (auto& pair : op->attributes()) {
    if (pair.first == name) {
      found = true;
      if (pair.second.isa<pir::DoubleAttribute>()) {
        *res = pair.second.dyn_cast<::pir::DoubleAttribute>().data();
        break;
      }
    }
  }
  PADDLE_ENFORCE_EQ(
      found,
      true,
      common::errors::InvalidArgument(
          "Cannot found attribute %s in op %s", name, op->name()));
}

void PaddlePirParser::GetOpAttr(const pir::Operation* op,
                                const std::string& name,
                                bool* res) const {
  bool found = false;
  for (auto& pair : op->attributes()) {
    if (pair.first == name) {
      found = true;
      if (pair.second.isa<pir::BoolAttribute>()) {
        *res = pair.second.dyn_cast<::pir::BoolAttribute>().data();
        break;
      }
    }
  }
  PADDLE_ENFORCE_EQ(
      found,
      true,
      common::errors::InvalidArgument(
          "Cannot found attribute %s in op %s", name, op->name()));
}

void PaddlePirParser::GetOpAttr(const pir::Operation* op,
                                const std::string& name,
                                std::string* res) const {
  bool found = false;
  for (auto& pair : op->attributes()) {
    if (pair.first == name) {
      found = true;
      if (pair.second.isa<pir::StrAttribute>()) {
        *res = pair.second.dyn_cast<::pir::StrAttribute>().AsString();
        break;
      }
    }
  }
  PADDLE_ENFORCE_EQ(
      found,
      true,
      common::errors::InvalidArgument(
          "Cannot found attribute %s in op %s", name, op->name()));
}

void PaddlePirParser::GetOpAttr(const pir::Operation* op,
                                const std::string& name,
                                std::vector<int64_t>* res) const {
  bool found = false;
  for (auto& pair : op->attributes()) {
    if (pair.first == name) {
      found = true;
      if (pair.second.isa<pir::ArrayAttribute>()) {
        auto array_list =
            pair.second.dyn_cast<::pir::ArrayAttribute>().AsVector();
        if (array_list.size() > 0) {
          // TODO(qzylalala): Need double check.
          PADDLE_ENFORCE_EQ(
              array_list[0].isa<::pir::Int64Attribute>() ||
                  array_list[0].isa<::pir::Int32Attribute>(),
              true,
              ::common::errors::Unimplemented("the 0th elementwise MUST be "
                                              "ir::Int64Attribute"));
          for (size_t i = 0; i < array_list.size(); ++i) {
            if (array_list[0].isa<::pir::Int64Attribute>()) {
              res->push_back(
                  array_list[i].dyn_cast<::pir::Int64Attribute>().data());
            } else {
              res->push_back(
                  array_list[i].dyn_cast<::pir::Int32Attribute>().data());
            }
          }
        }
      } else if (pair.second.isa<paddle::dialect::IntArrayAttribute>()) {
        *res = pair.second.dyn_cast<paddle::dialect::IntArrayAttribute>()
                   .data()
                   .GetData();
      }
      break;
    }
  }
  PADDLE_ENFORCE_EQ(
      found,
      true,
      common::errors::InvalidArgument(
          "Cannot found attribute %s in op %s", name, op->name()));
}

void PaddlePirParser::GetOpAttr(const pir::Operation* op,
                                const std::string& name,
                                std::vector<float>* res) const {
  bool found = false;
  for (auto& pair : op->attributes()) {
    if (pair.first == name) {
      found = true;
      if (pair.second.isa<pir::ArrayAttribute>()) {
        auto array_list =
            pair.second.dyn_cast<::pir::ArrayAttribute>().AsVector();
        if (array_list.size() > 0) {
          PADDLE_ENFORCE_EQ(
              array_list[0].isa<::pir::FloatAttribute>(),
              true,
              ::common::errors::Unimplemented("the 0th elementwise MUST be "
                                              "ir::FloatAttribute"));
          for (size_t i = 0; i < array_list.size(); ++i) {
            res->push_back(
                array_list[i].dyn_cast<::pir::FloatAttribute>().data());
          }
        }

        break;
      }
    }
  }
  PADDLE_ENFORCE_EQ(
      found,
      true,
      common::errors::InvalidArgument(
          "Cannot found attribute %s in op %s", name, op->name()));
}

void PaddlePirParser::GetOpAttr(const pir::Operation* op,
                                const std::string& name,
                                std::vector<double>* res) const {
  bool found = false;
  for (auto& pair : op->attributes()) {
    if (pair.first == name) {
      found = true;
      if (pair.second.isa<pir::ArrayAttribute>()) {
        auto array_list =
            pair.second.dyn_cast<::pir::ArrayAttribute>().AsVector();
        if (array_list.size() > 0) {
          PADDLE_ENFORCE_EQ(
              array_list[0].isa<::pir::DoubleAttribute>(),
              true,
              ::common::errors::Unimplemented("the 0th elementwise MUST be "
                                              "ir::DoubleAttribute"));
          for (size_t i = 0; i < array_list.size(); ++i) {
            res->push_back(
                array_list[i].dyn_cast<::pir::DoubleAttribute>().data());
          }
        }

        break;
      }
    }
  }
  PADDLE_ENFORCE_EQ(
      found,
      true,
      common::errors::InvalidArgument(
          "Cannot found attribute %s in op %s", name, op->name()));
}

void PaddlePirParser::GetOpAttr(const pir::Operation* op,
                                const std::string& name,
                                std::vector<bool>* res) const {
  bool found = false;
  for (auto& pair : op->attributes()) {
    if (pair.first == name) {
      found = true;
      if (pair.second.isa<pir::ArrayAttribute>()) {
        auto array_list =
            pair.second.dyn_cast<::pir::ArrayAttribute>().AsVector();
        if (array_list.size() > 0) {
          PADDLE_ENFORCE_EQ(
              array_list[0].isa<::pir::BoolAttribute>(),
              true,
              ::common::errors::Unimplemented("the 0th elementwise MUST be "
                                              "ir::BoolAttribute"));
          for (size_t i = 0; i < array_list.size(); ++i) {
            res->push_back(
                array_list[i].dyn_cast<::pir::BoolAttribute>().data());
          }
        }

        break;
      }
    }
  }
  PADDLE_ENFORCE_EQ(
      found,
      true,
      common::errors::InvalidArgument(
          "Cannot found attribute %s in op %s", name, op->name()));
}

void PaddlePirParser::GetOpScalarValue(int64_t op_id,
                                       bool if_in_sub_block,
                                       const std::string& scalar_attr_name,
                                       ScalarData* scalar_data) const {
  pir::Operation* op =
      if_in_sub_block ? sub_blocks_ops[op_id] : global_blocks_ops[op_id];
  PADDLE_ENFORCE_EQ(
      OpHasAttr(op, scalar_attr_name),
      true,
      common::errors::InvalidArgument(
          "Cannot found attribute %s in op %s", scalar_attr_name, op->name()));
  auto attr = op->attribute(scalar_attr_name);
  if (attr.isa<pir::DoubleAttribute>()) {
    *scalar_data =
        static_cast<double>(attr.dyn_cast<::pir::DoubleAttribute>().data());
  } else if (attr.isa<pir::FloatAttribute>()) {
    *scalar_data =
        static_cast<float>(attr.dyn_cast<::pir::FloatAttribute>().data());
  } else if (attr.isa<pir::Int64Attribute>()) {
    *scalar_data =
        static_cast<int64_t>(attr.dyn_cast<::pir::Int64Attribute>().data());
  } else if (attr.isa<pir::Int32Attribute>()) {
    *scalar_data =
        static_cast<int32_t>(attr.dyn_cast<::pir::Int32Attribute>().data());
  } else if (attr.isa<pir::BoolAttribute>()) {
    *scalar_data =
        static_cast<bool>(attr.dyn_cast<::pir::BoolAttribute>().data());
  } else {
    Assert(false,
           "ScalarData only support double, float, int64_t, int32_t and bool "
           "now.");
  }
}
std::vector<TensorInfo> PaddlePirParser::GetOpInput(
    int64_t op_id, int64_t input_idx, bool if_in_sub_block) const {
  PADDLE_ENFORCE_GT(input_idx,
                    -1,
                    common::errors::InvalidArgument(
                        "input_idx should be greater than -1 in GetOpInput."));
  pir::Operation* op;
  if (if_in_sub_block) {
    op = sub_blocks_ops[op_id];
  } else {
    op = global_blocks_ops[op_id];
  }
  PADDLE_ENFORCE_LT(input_idx,
                    op->num_operands(),
                    common::errors::InvalidArgument(
                        "input index %d is out of range, the input size is %d",
                        input_idx,
                        op->num_operands()));
  if (if_in_sub_block) {
    return GetSubBlockValueTensorInfo(op->operand(input_idx).source());
  } else {
    return GetTensorInfo(op->operand(input_idx).source());
  }
}

std::vector<TensorInfo> PaddlePirParser::GetOpOutput(
    int64_t op_id, int64_t output_idx, bool if_in_sub_block) const {
  PADDLE_ENFORCE_GT(
      output_idx,
      -1,
      common::errors::InvalidArgument(
          "output_idx should be greater than -1 in GetOpOutput."));
  pir::Operation* op =
      if_in_sub_block ? sub_blocks_ops[op_id] : global_blocks_ops[op_id];
  PADDLE_ENFORCE_LT(
      output_idx,
      op->num_results(),
      common::errors::InvalidArgument(
          "output index %d is out of range, the output size is %d",
          output_idx,
          op->num_results()));
  return if_in_sub_block ? GetSubBlockValueTensorInfo(op->result(output_idx))
                         : GetTensorInfo(op->result(output_idx));
}

/**
std::vector<int64_t> PaddlePirParser::GetOpAttrVar(int64_t op_id, int64_t
input_idx, const std::string &name) const { pir::Operation* op =
global_blocks_ops[op_id]->operand(input_idx).source().defining_op();
  std::vector<int64_t> result;
  GetOpAttr(op, name, &result);
  for(auto i : result)
  {
    std::cout << "attr: " << i << std::endl;
  }
  return result;
}
*/

bool PaddlePirParser::IsConstantTensor(int64_t op_id,
                                       int64_t input_idx,
                                       bool if_in_sub_block) const {
  PADDLE_ENFORCE_GT(
      input_idx,
      -1,
      common::errors::InvalidArgument(
          "input_idx should be greater than -1 in IsConstantTensor."));
  // todo(wangmingkai02): need to check
  pir::Operation* op =
      if_in_sub_block ? sub_blocks_ops[op_id] : global_blocks_ops[op_id];
  return op->operand(input_idx).source().defining_op()->num_operands() == 0 ||
         op->operand(input_idx).source().defining_op()->name() ==
             "pd_op.assign_value_";
}

void PaddlePirParser::SetTensorArrayName(int64_t op_id,
                                         bool if_in_sub_block,
                                         std::string tensor_arr_name) const {
  pir::Operation* op =
      if_in_sub_block ? sub_blocks_ops[op_id] : global_blocks_ops[op_id];
  if (op->num_operands() > 0) {
    op = op->operand(0).source().defining_op();
  }
  _tensor_arr_mappings[op] = tensor_arr_name;
}
std::string PaddlePirParser::GetTensorArrayName(int64_t op_id,
                                                bool if_in_sub_block) const {
  pir::Operation* temp_op =
      if_in_sub_block ? sub_blocks_ops[op_id] : global_blocks_ops[op_id];
  pir::Operation* op = temp_op->operand(0).source().defining_op();
  PADDLE_ENFORCE_EQ(
      _tensor_arr_mappings.count(op) > 0,
      true,
      common::errors::InvalidArgument(
          "Cannot find the tensor array used by op %s.", temp_op->name()));
  return _tensor_arr_mappings.at(op);
}

P2ODataType PaddlePirParser::TransPirDataType2OldIrDataType(
    phi::DataType dtype) const {
  // TODO(wangmingkai02): This is for compatibility with conversions under the
  // old IR, where P2ODataType is stored in TensorInfo.dtype. In the mapper,
  // GetOnnxDtype is called to obtain TensorProto_DataType in ONNX.
  // Add more type mappings if necessary.
  if (dtype == phi::DataType::UNDEFINED) {
    return P2ODataType::UNDEFINED;
  } else if (dtype == phi::DataType::BOOL) {
    return P2ODataType::BOOL;
  } else if (dtype == phi::DataType::UINT8) {
    return P2ODataType::UINT8;
  } else if (dtype == phi::DataType::INT8) {
    return P2ODataType::INT8;
  } else if (dtype == phi::DataType::UINT16) {
    return P2ODataType::INT16;
  } else if (dtype == phi::DataType::INT32) {
    return P2ODataType::INT32;
  } else if (dtype == phi::DataType::INT64) {
    return P2ODataType::INT64;
  } else if (dtype == phi::DataType::FLOAT32) {
    return P2ODataType::FP32;
  } else if (dtype == phi::DataType::FLOAT64) {
    return P2ODataType::FP64;
  } else if (dtype == phi::DataType::COMPLEX64) {
    return P2ODataType::COMPLEX64;
  } else if (dtype == phi::DataType::COMPLEX128) {
    return P2ODataType::COMPLEX128;
  } else if (dtype == phi::DataType::FLOAT16) {
    return P2ODataType::FP16;
  } else if (dtype == phi::DataType::BFLOAT16) {
    return P2ODataType::BF16;
  } else {
    Assert(false,
           "Unsupported data type exists in "
           "PaddlePirParser::TransPirDataType2OnnxDataType.");
  }
}
void PaddlePirParser::GetWhileInputValuesAndArgsMappings(
    paddle::dialect::WhileOp* while_op) const {
  // mapping args and inputs in while op using while_op_values_args_map
  std::vector<pir::detail::ValueImpl*> while_op_input_value_address;
  std::vector<pir::detail::ValueImpl*> while_op_input_arg_address;
  // record input value address
  for (int index = 1; index < while_op->num_operands(); index++) {
    const pir::Value& value = while_op->operand_source(index);
    while_op_input_value_address.push_back(
        &(*(value).impl()));  // get value address
  }
  // record args value address
  std::vector<pir::Value> args = while_op->block_args();
  for (int i = 0; i < args.size(); i++) {
    const pir::Value& value = args[i];
    while_op_input_arg_address.push_back(&(*(value.impl())));
  }

  // mapping
  for (int index = 0; index < while_op_input_value_address.size(); index++) {
    auto arg_addr = while_op_input_arg_address[index];
    if (while_op_values_args_map.count(arg_addr)) continue;
    auto value_addr = while_op_input_value_address[index];
    while (while_op_values_args_map.count(value_addr)) {
      value_addr = while_op_values_args_map[value_addr];
    }
    while_op_values_args_map[arg_addr] = value_addr;
  }
}

}  // namespace paddle2onnx
