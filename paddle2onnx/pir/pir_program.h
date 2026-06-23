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
//
// Standalone PIR IR types — JSON-backed, no dependency on libpaddle.so.
// Provides a minimal subset of the PIR API that paddle2onnx actually uses.

#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "nlohmann/json.hpp"

namespace paddle2onnx {
namespace pir {

// ---------------------------------------------------------------------------
// Forward declarations / lightweight data types
// ---------------------------------------------------------------------------

using Json = nlohmann::json;

// ---------------------------------------------------------------------------
// Data types
// ---------------------------------------------------------------------------
enum class DataType {
  UNDEFINED = 0,
  FLOAT32 = 1,
  FLOAT64 = 2,
  FLOAT16 = 3,
  BFLOAT16 = 4,
  INT32 = 5,
  INT64 = 6,
  INT16 = 7,
  INT8 = 8,
  UINT8 = 9,
  BOOL = 10,
  COMPLEX64 = 11,
  COMPLEX128 = 12,
  FLOAT8_E4M3FN = 13,
  FLOAT8_E5M2 = 14,
};

DataType DataTypeFromString(const std::string& s);
std::string DataTypeToString(DataType dt);

// ---------------------------------------------------------------------------
// Type — minimal type info extracted from JSON
// ---------------------------------------------------------------------------
class Type {
 public:
  Type() = default;
  explicit Type(const Json& j) { Parse(j); }

  void Parse(const Json& j);

  DataType dtype() const { return dtype_; }
  const std::vector<int64_t>& dims() const { return dims_; }
  bool is_dense_tensor() const { return is_dense_tensor_; }
  bool is_tensor_array() const { return is_tensor_array_; }

 private:
  DataType dtype_ = DataType::UNDEFINED;
  std::vector<int64_t> dims_;
  bool is_dense_tensor_ = false;
  bool is_tensor_array_ = false;
};

// ---------------------------------------------------------------------------
// Attribute types — type tags for isa/dyn_cast
// ---------------------------------------------------------------------------
enum class AttrType {
  kInt32,
  kInt64,
  kFloat,
  kDouble,
  kBool,
  kString,
  kArray,
  kUndefined,
};

// ---------------------------------------------------------------------------
// Attribute — attribute value backed by JSON
// ---------------------------------------------------------------------------
class Attribute {
 public:
  Attribute() = default;
  Attribute(const Json& j, const std::string& name = "")
      : json_(&j), name_(name) {
    if (json_) DetectType();
  }

  const std::string& name() const { return name_; }
  bool valid() const { return json_ != nullptr; }

  // Type checking
  AttrType type() const { return type_; }
  template <typename T>
  bool isa() const;
  template <typename T>
  T dyn_cast() const;

  // Typed accessors
  int32_t AsInt32() const;
  int64_t AsInt64() const;
  float AsFloat() const;
  double AsDouble() const;
  bool AsBool() const;
  std::string AsString() const;

  // Array accessors
  std::vector<int32_t> AsInt32Array() const;
  std::vector<int64_t> AsInt64Array() const;
  std::vector<float> AsFloatArray() const;
  std::vector<double> AsDoubleArray() const;
  std::vector<bool> AsBoolArray() const;
  std::vector<Attribute> AsVector() const;

  // Raw data access
  const Json& raw_json() const { return *json_; }

 private:
  void DetectType();
  const Json* json_ = nullptr;
  std::string name_;
  AttrType type_ = AttrType::kUndefined;
};

// Attribute type trait specializations for isa/dyn_cast
class Int32Attribute;
class Int64Attribute;
class FloatAttribute;
class DoubleAttribute;
class BoolAttribute;
class StrAttribute;
class ArrayAttribute;

template <>
inline bool Attribute::isa<Int32Attribute>() const {
  return type_ == AttrType::kInt32;
}
template <>
inline bool Attribute::isa<Int64Attribute>() const {
  return type_ == AttrType::kInt64;
}
template <>
inline bool Attribute::isa<FloatAttribute>() const {
  return type_ == AttrType::kFloat;
}
template <>
inline bool Attribute::isa<DoubleAttribute>() const {
  return type_ == AttrType::kDouble;
}
template <>
inline bool Attribute::isa<BoolAttribute>() const {
  return type_ == AttrType::kBool;
}
template <>
inline bool Attribute::isa<StrAttribute>() const {
  return type_ == AttrType::kString;
}
template <>
inline bool Attribute::isa<ArrayAttribute>() const {
  return type_ == AttrType::kArray;
}

// ---------------------------------------------------------------------------
// Concrete attribute subclasses (thin wrappers for dyne_cast compatibility)
// ---------------------------------------------------------------------------
class Int32Attribute : public Attribute {
 public:
  using Attribute::Attribute;
  int32_t data() const { return AsInt32(); }
};
class Int64Attribute : public Attribute {
 public:
  using Attribute::Attribute;
  int64_t data() const { return AsInt64(); }
};
class FloatAttribute : public Attribute {
 public:
  using Attribute::Attribute;
  float data() const { return AsFloat(); }
};
class DoubleAttribute : public Attribute {
 public:
  using Attribute::Attribute;
  double data() const { return AsDouble(); }
};
class BoolAttribute : public Attribute {
 public:
  using Attribute::Attribute;
  bool data() const { return AsBool(); }
};
class StrAttribute : public Attribute {
 public:
  using Attribute::Attribute;
  std::string AsString() const { return Attribute::AsString(); }
};
class ArrayAttribute : public Attribute {
 public:
  using Attribute::Attribute;
  std::vector<Attribute> AsVector() const { return Attribute::AsVector(); }
};

// Template specialization for dyn_cast
template <>
inline Int32Attribute Attribute::dyn_cast<Int32Attribute>() const {
  return Int32Attribute(*json_, name_);
}
template <>
inline Int64Attribute Attribute::dyn_cast<Int64Attribute>() const {
  return Int64Attribute(*json_, name_);
}
template <>
inline FloatAttribute Attribute::dyn_cast<FloatAttribute>() const {
  return FloatAttribute(*json_, name_);
}
template <>
inline DoubleAttribute Attribute::dyn_cast<DoubleAttribute>() const {
  return DoubleAttribute(*json_, name_);
}
template <>
inline BoolAttribute Attribute::dyn_cast<BoolAttribute>() const {
  return BoolAttribute(*json_, name_);
}
template <>
inline StrAttribute Attribute::dyn_cast<StrAttribute>() const {
  return StrAttribute(*json_, name_);
}
template <>
inline ArrayAttribute Attribute::dyn_cast<ArrayAttribute>() const {
  return ArrayAttribute(*json_, name_);
}

// Catch-all: if T is not one of the known types, return empty
template <typename T>
inline T Attribute::dyn_cast() const {
  return T();
}

// Forward declarations
class Operation;
class Block;
class Value;
class OpResult;

// ---------------------------------------------------------------------------
// Value — a value (result of some op, or block argument)
// ---------------------------------------------------------------------------
class Value {
 public:
  Value() = default;
  Value(int64_t id, const Type& type = Type())
      : id_(id), type_(type) {}

  int64_t id() const { return id_; }
  const Type& type() const { return type_; }

  // defining_op — requires external lookup, returns nullptr by default.
  // Set after parsing via set_defining_op.
  Operation* defining_op() const { return defining_op_; }
  void set_defining_op(Operation* op, int64_t result_idx = 0) {
    defining_op_ = op;
    result_idx_ = result_idx;
  }
  int64_t result_index() const { return result_idx_; }

 private:
  int64_t id_ = -1;
  Type type_;
  Operation* defining_op_ = nullptr;
  int64_t result_idx_ = 0;
};

// ---------------------------------------------------------------------------
// OpOperand — references an input value
// ---------------------------------------------------------------------------
struct OpOperand {
  int64_t value_id = -1;  // reference to a value ID
  Value resolved_source;  // resolved value (set during parsing)

  // Convenience: return resolved Value
  const Value& source() const { return resolved_source; }
  Value& source() { return resolved_source; }
};

// ---------------------------------------------------------------------------
// OpResult — one output of an operation (inherits Value)
// ---------------------------------------------------------------------------
class OpResult : public Value {
 public:
  OpResult() = default;
  OpResult(int64_t id, const Type& type, Operation* defining_op = nullptr,
           int64_t index = 0)
      : Value(id, type) {
    set_defining_op(defining_op, index);
  }

  int64_t index() const { return result_index(); }
};

// ---------------------------------------------------------------------------
// Operation — a single operation in the PIR graph
// ---------------------------------------------------------------------------
class Operation {
 public:
  Operation() = default;

  const std::string& name() const { return name_; }
  void set_name(const std::string& n) { name_ = n; }

  size_t num_operands() const { return operands_.size(); }
  const OpOperand& operand(size_t i) const { return operands_[i]; }
  void add_operand(const OpOperand& op) { operands_.push_back(op); }
  // Return the source Value of the i-th operand
  Value operand_source(size_t i) const { return operand(i).source(); }

  size_t num_results() const { return results_.size(); }
  const OpResult& result(size_t i) const { return results_[i]; }
  void add_result(const OpResult& r) { results_.push_back(r); }

  // Parser identity
  void set_id(int64_t id) { id_ = id; }
  int64_t id() const { return id_; }

  // Parent block
  Block* GetParent() const { return parent_; }
  void set_parent(Block* b) { parent_ = b; }

  bool HasAttribute(const std::string& name) const;
  Attribute attribute(const std::string& name) const;
  const std::map<std::string, Attribute>& attributes() const {
    return attrs_;
  }
  void add_attribute(const std::string& name, const Attribute& attr);

  size_t num_regions() const { return sub_blocks_.size(); }
  class Block& region(size_t i) { return sub_blocks_[i]; }
  const class Block& region(size_t i) const { return sub_blocks_[i]; }
  void add_region(const class Block& b) { sub_blocks_.push_back(b); }

 private:
  std::string name_;
  int64_t id_ = -1;
  Block* parent_ = nullptr;
  std::vector<OpOperand> operands_;
  std::vector<OpResult> results_;
  std::map<std::string, Attribute> attrs_;
  std::vector<class Block> sub_blocks_;
};

// ---------------------------------------------------------------------------
// Block — contains args and ops
// ---------------------------------------------------------------------------
class Block {
 public:
  Block() = default;

  const std::vector<Value>& args() const { return args_; }
  void add_arg(const Value& v) { args_.push_back(v); }

  // ops() returns non-const references to allow mutation during parsing
  std::vector<Operation>& ops() { return ops_; }
  const std::vector<Operation>& ops() const { return ops_; }
  void add_op(const Operation& op) { ops_.push_back(op); }

 private:
  std::vector<Value> args_;
  std::vector<Operation> ops_;
};

// ---------------------------------------------------------------------------
// Region — contains blocks
// ---------------------------------------------------------------------------
class Region {
 public:
  Region() = default;
  std::vector<Block>& blocks() { return blocks_; }
  const std::vector<Block>& blocks() const { return blocks_; }
  void add_block(const Block& b) { blocks_.push_back(b); }

 private:
  std::vector<Block> blocks_;
};

// ---------------------------------------------------------------------------
// Program — top-level container
// ---------------------------------------------------------------------------
class Program {
 public:
  Program() = default;

  // Load from JSON string
  bool LoadFromJson(const std::string& json_str);

  // Accessors
  Block* block();
  const Block* block() const;
  const std::vector<Region>& regions() const { return regions_; }

  // For compatibility with pir::Program interface
  class ModuleOp {
   public:
    ModuleOp() = default;
    size_t num_regions() const { return 0; }
    class Region& region(size_t) {
      static Region dummy;
      return dummy;
    }
  };
  ModuleOp module_op() const { return ModuleOp(); }

 private:
  std::shared_ptr<Json> root_;
  std::vector<Region> regions_;
  bool ParseOp(Operation* op, const Json& j);
  bool ParseBlock(Block* block, const Json& j);
  bool ParseRegion(Region* region, const Json& j);
  void DecompressOpName(std::string* name) const;
  void ResolveValues();
};

// ---------------------------------------------------------------------------
// Helper: lookup value by ID (across the whole program)
// ---------------------------------------------------------------------------
class ValueTable {
 public:
  void Add(const Value& v) { map_[v.id()] = v; }
  const Value* Find(int64_t id) const {
    auto it = map_.find(id);
    return it != map_.end() ? &it->second : nullptr;
  }

 private:
  std::map<int64_t, Value> map_;
};

// ---------------------------------------------------------------------------
// Dialect ID map for compressed op names
// ---------------------------------------------------------------------------
class DialectIdMap {
 public:
  static DialectIdMap* Instance();
  DialectIdMap();
  std::string DecompressOpName(const std::string& compressed) const;

 private:
  std::unordered_map<std::string, std::string> decompress_map_;
};

}  // namespace pir
}  // namespace paddle2onnx
