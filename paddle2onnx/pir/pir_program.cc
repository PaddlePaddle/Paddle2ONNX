// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle2onnx/pir/pir_program.h"

#include <fstream>
#include <iostream>
#include <sstream>

namespace paddle2onnx {
namespace pir {

// ===========================================================================
// Data type converters
// ===========================================================================
DataType DataTypeFromString(const std::string& s) {
  if (s == "bfloat16" || s == "BF16") return DataType::BFLOAT16;
  if (s == "float16" || s == "f16" || s == "FLOAT16") return DataType::FLOAT16;
  if (s == "float32" || s == "f32" || s == "FLOAT32") return DataType::FLOAT32;
  if (s == "float64" || s == "f64" || s == "FLOAT64") return DataType::FLOAT64;
  if (s == "uint8" || s == "UINT8") return DataType::UINT8;
  if (s == "int8" || s == "INT8") return DataType::INT8;
  if (s == "int16" || s == "INT16") return DataType::INT16;
  if (s == "int32" || s == "INT32") return DataType::INT32;
  if (s == "int64" || s == "INT64") return DataType::INT64;
  if (s == "bool" || s == "BOOL") return DataType::BOOL;
  if (s == "complex64" || s == "COMPLEX64") return DataType::COMPLEX64;
  if (s == "complex128" || s == "COMPLEX128") return DataType::COMPLEX128;
  return DataType::UNDEFINED;
}

std::string DataTypeToString(DataType dt) {
  switch (dt) {
    case DataType::FLOAT32:
      return "float32";
    case DataType::FLOAT64:
      return "float64";
    case DataType::FLOAT16:
      return "float16";
    case DataType::BFLOAT16:
      return "bfloat16";
    case DataType::INT32:
      return "int32";
    case DataType::INT64:
      return "int64";
    case DataType::INT16:
      return "int16";
    case DataType::INT8:
      return "int8";
    case DataType::UINT8:
      return "uint8";
    case DataType::BOOL:
      return "bool";
    default:
      return "undefined";
  }
}

// ===========================================================================
// Type — parse from JSON type descriptor
// ===========================================================================
void Type::Parse(const Json& j) {
  if (j.is_null() || j.empty()) return;

  // PIR type JSON can be: {"#": "builtin.tensor<3x4xf32>"} or {"TT": "f32",
  // "D": [3,4]} Note: "#" in block arg context is a value ID (number), not a
  // type string.
  std::string type_str;
  if (j.is_string()) {
    type_str = j.get<std::string>();
  } else if (j.contains("TT") && j["TT"].is_string()) {
    type_str = j["TT"].get<std::string>();
  } else if (j.contains("#") && j["#"].is_string()) {
    type_str = j["#"].get<std::string>();
  }

  if (type_str.empty()) return;

  // Check if it's a dense tensor type: builtin.tensor<...>
  if (type_str.find("tensor<") != std::string::npos ||
      type_str.find("Tensor<") != std::string::npos) {
    is_dense_tensor_ = true;
    // Extract dtype and shape from string like "builtin.tensor<3x4xf32>"
    auto start = type_str.find('<');
    auto end = type_str.rfind('>');
    if (start != std::string::npos && end != std::string::npos) {
      std::string inner = type_str.substr(start + 1, end - start - 1);
      // Split by 'x' - last part is dtype, rest are dims
      auto last_x = inner.rfind('x');
      if (last_x != std::string::npos) {
        std::string dtype_str = inner.substr(last_x + 1);
        dtype_ = DataTypeFromString(dtype_str);
        std::string dims_str = inner.substr(0, last_x);
        // Parse dimensions separated by 'x'
        size_t pos = 0;
        while (pos < dims_str.size()) {
          size_t next = dims_str.find('x', pos);
          std::string dim_str = dims_str.substr(pos, next - pos);
          if (!dim_str.empty() && dim_str != "S") {  // S = dynamic
            try {
              dims_.push_back(std::stoll(dim_str));
            } catch (...) {
              dims_.push_back(-1);  // dynamic dim
            }
          } else {
            dims_.push_back(-1);  // dynamic dim
          }
          if (next == std::string::npos) break;
          pos = next + 1;
        }
      }
    }
  } else if (type_str == "f32" || type_str == "float32") {
    dtype_ = DataType::FLOAT32;
    is_dense_tensor_ = true;
  } else if (type_str == "f64" || type_str == "float64") {
    dtype_ = DataType::FLOAT64;
    is_dense_tensor_ = true;
  } else if (type_str == "i32" || type_str == "int32") {
    dtype_ = DataType::INT32;
    is_dense_tensor_ = true;
  } else if (type_str == "i64" || type_str == "int64") {
    dtype_ = DataType::INT64;
    is_dense_tensor_ = true;
  } else if (type_str == "bool") {
    dtype_ = DataType::BOOL;
    is_dense_tensor_ = true;
  }

  // Also try to parse dims from JSON array in "D" field
  if (j.contains("D") && j["D"].is_array()) {
    dims_.clear();
    for (const auto& d : j["D"]) {
      dims_.push_back(d.is_number() ? d.get<int64_t>() : -1);
    }
  }
}

// ===========================================================================
// Attribute — type detection
// ===========================================================================
void Attribute::DetectType() {
  if (!json_) return;
  auto& j = *json_;
  if (!j.contains("AT") && !j.contains("N")) {
    // For simple values without type annotation, guess from content
    if (j.is_boolean())
      type_ = AttrType::kBool;
    else if (j.is_number_integer())
      type_ = AttrType::kInt64;
    else if (j.is_number_float())
      type_ = AttrType::kDouble;
    else if (j.is_string())
      type_ = AttrType::kString;
    return;
  }

  std::string at;
  if (j.contains("AT")) {
    if (j["AT"].is_string())
      at = j["AT"].get<std::string>();
    else if (j["AT"].is_array() && !j["AT"].empty() && j["AT"][0].is_string())
      at = j["AT"][0].get<std::string>();
  }

  if (at.find("Int32") != std::string::npos)
    type_ = AttrType::kInt32;
  else if (at.find("Int64") != std::string::npos)
    type_ = AttrType::kInt64;
  else if (at.find("Float") != std::string::npos ||
           at.find("float") != std::string::npos) {
    if (at.find("64") != std::string::npos ||
        at.find("double") != std::string::npos)
      type_ = AttrType::kDouble;
    else
      type_ = AttrType::kFloat;
  } else if (at.find("Bool") != std::string::npos)
    type_ = AttrType::kBool;
  else if (at.find("Str") != std::string::npos)
    type_ = AttrType::kString;
  else if (at.find("Array") != std::string::npos ||
           at.find("array") != std::string::npos) {
    type_ = AttrType::kArray;
    // Check element type
    if (j.contains("D") && j["D"].is_array() && !j["D"].empty()) {
      // Detect array element type from first element
    }
  }
  // Fallback: check if D is an array (probably an array attribute)
  if (type_ == AttrType::kUndefined && j.contains("D") && j["D"].is_array()) {
    type_ = AttrType::kArray;
  }
}

// ===========================================================================
// Attribute — typed accessors for JSON-backed attributes
// ===========================================================================
// Attribute JSON format: {"N": "attr_name", "AT": "type_descriptor", "D":
// [...]} or simpler: ["attr_name", value]
int32_t Attribute::AsInt32() const {
  if (!json_) return 0;
  auto& j = *json_;
  if (j.contains("D") && j["D"].is_array() && !j["D"].empty()) {
    auto& d = j["D"][0];
    if (d.is_number()) return d.get<int32_t>();
  }
  if (j.contains("D") && j["D"].is_number()) return j["D"].get<int32_t>();
  if (j.is_number()) return j.get<int32_t>();
  return 0;
}

int64_t Attribute::AsInt64() const {
  if (!json_) return 0;
  auto& j = *json_;
  if (j.contains("D") && j["D"].is_array() && !j["D"].empty()) {
    auto& d = j["D"][0];
    if (d.is_number()) return d.get<int64_t>();
  }
  if (j.contains("D") && j["D"].is_number()) return j["D"].get<int64_t>();
  if (j.is_number()) return j.get<int64_t>();
  return 0;
}

float Attribute::AsFloat() const {
  if (!json_) return 0.0f;
  auto& j = *json_;
  if (j.contains("D") && j["D"].is_array() && !j["D"].empty()) {
    auto& d = j["D"][0];
    if (d.is_number()) return d.get<float>();
  }
  if (j.contains("D") && j["D"].is_number()) return j["D"].get<float>();
  if (j.is_number()) return j.get<float>();
  return 0.0f;
}

double Attribute::AsDouble() const {
  if (!json_) return 0.0;
  auto& j = *json_;
  if (j.contains("D") && j["D"].is_array() && !j["D"].empty()) {
    auto& d = j["D"][0];
    if (d.is_number_float()) return d.get<double>();
    return d.get<double>();
  }
  if (j.contains("D") && j["D"].is_number()) return j["D"].get<double>();
  if (j.is_number()) return j.get<double>();
  return 0.0;
}

bool Attribute::AsBool() const {
  if (!json_) return false;
  auto& j = *json_;
  if (j.contains("D") && j["D"].is_array() && !j["D"].empty()) {
    auto& d = j["D"][0];
    if (d.is_boolean()) return d.get<bool>();
    return d.get<int32_t>() != 0;
  }
  if (j.contains("D") && j["D"].is_boolean()) return j["D"].get<bool>();
  if (j.is_boolean()) return j.get<bool>();
  if (j.is_number()) return j.get<int32_t>() != 0;
  return false;
}

std::string Attribute::AsString() const {
  if (!json_) return "";
  auto& j = *json_;
  if (j.contains("D") && j["D"].is_array() && !j["D"].empty()) {
    auto& d = j["D"][0];
    if (d.is_string()) return d.get<std::string>();
  }
  if (j.contains("D") && j["D"].is_string()) return j["D"].get<std::string>();
  if (j.is_string()) return j.get<std::string>();
  return "";
}

std::vector<int32_t> Attribute::AsInt32Array() const {
  std::vector<int32_t> result;
  if (!json_) return result;
  auto& j = *json_;
  if (j.contains("D") && j["D"].is_array()) {
    for (auto& v : j["D"]) {
      if (v.is_number()) result.push_back(v.get<int32_t>());
    }
  }
  return result;
}

std::vector<int64_t> Attribute::AsInt64Array() const {
  std::vector<int64_t> result;
  if (!json_) return result;
  auto& j = *json_;
  if (j.contains("D") && j["D"].is_array()) {
    for (auto& v : j["D"]) {
      if (v.is_number()) result.push_back(v.get<int64_t>());
    }
  }
  return result;
}

std::vector<float> Attribute::AsFloatArray() const {
  std::vector<float> result;
  if (!json_) return result;
  auto& j = *json_;
  if (j.contains("D") && j["D"].is_array()) {
    for (auto& v : j["D"]) {
      if (v.is_number()) result.push_back(v.get<float>());
    }
  }
  return result;
}

std::vector<double> Attribute::AsDoubleArray() const {
  std::vector<double> result;
  if (!json_) return result;
  auto& j = *json_;
  if (j.contains("D") && j["D"].is_array()) {
    for (auto& v : j["D"]) {
      if (v.is_number_float())
        result.push_back(v.get<double>());
      else
        result.push_back(static_cast<double>(v.get<float>()));
    }
  }
  return result;
}

std::vector<bool> Attribute::AsBoolArray() const {
  std::vector<bool> result;
  if (!json_) return result;
  auto& j = *json_;
  if (j.contains("D") && j["D"].is_array()) {
    for (auto& v : j["D"]) {
      if (v.is_boolean())
        result.push_back(v.get<bool>());
      else
        result.push_back(v.get<int32_t>() != 0);
    }
  }
  return result;
}

std::vector<Attribute> Attribute::AsVector() const {
  std::vector<Attribute> result;
  if (!json_) return result;
  auto& j = *json_;
  if (j.contains("D") && j["D"].is_array()) {
    for (auto& v : j["D"]) {
      // Create sub-attributes for each element
      Json elem;
      elem["D"] = Json::array({v});
      elem["N"] = "";
      elem["AT"] = "";
      result.emplace_back(elem);
    }
  }
  return result;
}

// ===========================================================================
// Operation
// ===========================================================================
bool Operation::HasAttribute(const std::string& name) const {
  return attrs_.find(name) != attrs_.end();
}

Attribute Operation::attribute(const std::string& name) const {
  auto it = attrs_.find(name);
  if (it != attrs_.end()) return it->second;
  return Attribute();
}

void Operation::add_attribute(const std::string& name, const Attribute& attr) {
  attrs_[name] = attr;
}

// ===========================================================================
// Program — parse JSON into our IR types
// ===========================================================================
bool Program::LoadFromJson(const std::string& json_str) {
  try {
    root_ = std::make_shared<Json>(Json::parse(json_str));
  } catch (const std::exception& e) {
    std::cerr << "[ERROR] Failed to parse PIR JSON: " << e.what() << std::endl;
    return false;
  }

  // PIR JSON structure:
  // { "program": { "regions": [{ "blocks": [{ "args": [...], "ops": [...] }] }]
  // }
  //     or  { "regions": [...] } (no "program" wrapper)
  Json* prog_root = root_.get();
  Json* program_json = nullptr;
  if (prog_root->contains("program")) {
    program_json = &(*prog_root)["program"];
  } else {
    program_json = prog_root;
  }

  if (!program_json->contains("regions")) {
    std::cerr << "[ERROR] PIR JSON has no 'regions' key" << std::endl;
    return false;
  }

  for (auto& region_j : (*program_json)["regions"]) {
    Region region;
    if (!ParseRegion(&region, region_j)) return false;
    regions_.push_back(std::move(region));
  }

  // Second pass: resolve value references (ID → OpResult)
  ResolveValues();

  return true;
}

bool Program::ParseRegion(Region* region, const Json& region_j) {
  if (!region_j.contains("blocks")) {
    std::cerr << "[ERROR] Region has no 'blocks'" << std::endl;
    return false;
  }
  for (auto& block_j : region_j["blocks"]) {
    Block block;
    if (!ParseBlock(&block, block_j)) return false;
    region->add_block(std::move(block));
  }
  return true;
}

bool Program::ParseBlock(Block* block, const Json& block_j) {
  // Parse block args
  if (block_j.contains("args")) {
    for (auto& arg_j : block_j["args"]) {
      Value v;
      if (arg_j.contains("#")) {
        v = Value(arg_j["#"].get<int64_t>(), Type(arg_j));
      } else if (arg_j.contains("%")) {
        v = Value(arg_j["%"].get<int64_t>(), Type(arg_j));
      }
      block->add_arg(v);
    }
  }

  // Parse ops
  if (block_j.contains("ops")) {
    for (auto& op_j : block_j["ops"]) {
      Operation op;
      if (!ParseOp(&op, op_j)) return false;
      block->add_op(std::move(op));
    }
  }

  return true;
}

bool Program::ParseOp(Operation* op, const Json& op_j) {
  // Op name — stored in "#" field (compressed or full)
  std::string op_name;
  if (op_j.contains("#") && op_j["#"].is_string()) {
    op_name = op_j["#"].get<std::string>();
  } else if (op_j.contains("op_type") && op_j["op_type"].is_string()) {
    op_name = op_j["op_type"].get<std::string>();
  }
  DecompressOpName(&op_name);
  op->set_name(op_name);

  // Parse operands (inputs) — "I" field
  if (op_j.contains("I")) {
    for (auto& inp_j : op_j["I"]) {
      OpOperand operand;
      if (inp_j.is_object() && inp_j.contains("%")) {
        operand.value_id = inp_j["%"].get<int64_t>();
      } else if (inp_j.is_number()) {
        operand.value_id = inp_j.get<int64_t>();
      }
      op->add_operand(operand);
    }
  }

  // Parse results (outputs) — "O" field
  if (op_j.contains("O")) {
    int64_t idx = 0;
    for (auto& out_j : op_j["O"]) {
      int64_t id = -1;
      Type type;
      if (out_j.is_object()) {
        if (out_j.contains("%")) id = out_j["%"].get<int64_t>();
        type = Type(out_j);
      } else if (out_j.is_number()) {
        id = out_j.get<int64_t>();
      }
      // Create OpResult with defining_op = this op
      op->add_result(OpResult(id, type, op, idx++));
    }
  }

  // Parse attributes — "A" field
  // Format: [{"N": "name", "AT": "type", "D": [data]}, ...]
  // Or for ParameterOp: [0, 1, 0, "param_name", ...]  (fixed array)
  if (op_j.contains("A") && op_j["A"].is_array()) {
    for (auto& attr_j : op_j["A"]) {
      if (attr_j.is_object() && attr_j.contains("N")) {
        std::string attr_name = attr_j["N"].get<std::string>();
        op->add_attribute(attr_name, Attribute(attr_j, attr_name));
      }
    }
  }

  // ParameterOp special handling: "A" field is a fixed-format array
  // ["is_distributed", "is_parameter", "need_clip", "parameter_name", ...]
  if (op_name == "builtin.parameter" && op_j.contains("A")) {
    auto& a_arr = op_j["A"];
    if (a_arr.is_array() && a_arr.size() >= 4) {
      // Index 3 = parameter_name
      if (a_arr[3].is_string()) {
        Attribute attr(a_arr[3], "parameter_name");
        op->add_attribute("parameter_name", attr);
      }
      // Index 4 = persistable (0/1)
      if (a_arr.size() >= 5 && a_arr[4].is_number()) {
        Json persistable_j;
        persistable_j["D"] = Json::array({a_arr[4]});
        op->add_attribute("persistable",
                          Attribute(persistable_j, "persistable"));
      }
      // Index 5 = stop_gradient
      if (a_arr.size() >= 6 && a_arr[5].is_number()) {
        Json sg_j;
        sg_j["D"] = Json::array({a_arr[5]});
        op->add_attribute("stop_gradient", Attribute(sg_j, "stop_gradient"));
      }
    }
  }

  // Parse sub-block ops (for if/while ops)
  if (op_j.contains("regions")) {
    for (auto& region_j : op_j["regions"]) {
      Block block;
      ParseBlock(&block, region_j);
      op->add_region(block);
    }
  }

  return true;
}

void Program::DecompressOpName(std::string* name) const {
  if (name->empty()) return;
  auto* dial_map = DialectIdMap::Instance();
  *name = dial_map->DecompressOpName(*name);
}

Block* Program::block() {
  if (regions_.empty()) return nullptr;
  if (regions_[0].blocks().empty()) return nullptr;
  return &regions_[0].blocks()[0];
}

const Block* Program::block() const {
  if (regions_.empty()) return nullptr;
  if (regions_[0].blocks().empty()) return nullptr;
  return &regions_[0].blocks()[0];
}

// ===========================================================================
// DialectIdMap — compressed op name to full name
// ===========================================================================
DialectIdMap* DialectIdMap::Instance() {
  static DialectIdMap instance;
  return &instance;
}

DialectIdMap::DialectIdMap() {
  // Number-based dialect IDs (from PaddlePaddle serialization)
  decompress_map_["0"] = "builtin";
  decompress_map_["1"] = "pd_op";
  decompress_map_["2"] = "cf";
  decompress_map_["3"] = "custom_op";
  decompress_map_["4"] = "dist";
  decompress_map_[".dispatch"] = ".dispatch";

  // Single-character compressed ops
  decompress_map_["p"] = "builtin.parameter";
  decompress_map_["c"] = "builtin.combine";
  decompress_map_["s"] = "builtin.set_parameter";
  decompress_map_["g"] = "builtin.get_parameter";
}

std::string DialectIdMap::DecompressOpName(
    const std::string& compressed) const {
  // Check if it's a compressed single-char op (like "p" = "builtin.parameter")
  auto it = decompress_map_.find(compressed);
  if (it != decompress_map_.end()) {
    return it->second;
  }

  // Check if it's already a full name (contains "." like "pd_op.conv2d")
  if (compressed.find('.') != std::string::npos) {
    // Split by first "." and check if the prefix is a compressed dialect ID
    auto dot_pos = compressed.find('.');
    std::string prefix = compressed.substr(0, dot_pos);
    std::string suffix = compressed.substr(dot_pos + 1);
    auto prefix_it = decompress_map_.find(prefix);
    if (prefix_it != decompress_map_.end()) {
      return prefix_it->second + "." + suffix;
    }
    // Already full name (e.g., "pd_op.conv2d")
    return compressed;
  }

  // Otherwise treat as "pd_op.{name}"
  return "pd_op." + compressed;
}

// ===========================================================================
// ResolveValues — second pass to resolve operand value references
// ===========================================================================
void Program::ResolveValues() {
  // Build map: value_id → OpResult*
  std::map<int64_t, OpResult*> value_map;
  for (auto& region : regions_) {
    for (auto& block : region.blocks()) {
      for (auto& op : const_cast<Block&>(block).ops()) {
        for (size_t i = 0; i < op.num_results(); ++i) {
          auto& r = const_cast<OpResult&>(op.result(i));
          value_map[r.id()] = &r;
        }
      }
    }
  }

  // Resolve each operand's value_id to the actual Value object
  for (auto& region : regions_) {
    for (auto& block : region.blocks()) {
      for (auto& op : const_cast<Block&>(block).ops()) {
        for (size_t i = 0; i < op.num_operands(); ++i) {
          auto& operand = const_cast<OpOperand&>(op.operand(i));
          auto it = value_map.find(operand.value_id);
          if (it != value_map.end()) {
            operand.resolved_source = *it->second;
          }
        }
      }
    }
  }
}

}  // namespace pir
}  // namespace paddle2onnx
