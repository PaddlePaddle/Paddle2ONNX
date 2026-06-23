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
// Op name compatibility and info lookup — standalone YAML-backed.
// No dependency on libpaddle.so or OpRegistry.

#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace paddle2onnx {
namespace pir {

// ---------------------------------------------------------------------------
// OpNameNormalizer — maps between old (fluid) and new (phi) op names,
// and maps between op arg names (PIR internal → legacy names).
// Data extracted from PaddlePaddle's op_compat.yaml.
// ---------------------------------------------------------------------------
class OpNameNormalizer {
 public:
  static OpNameNormalizer* Instance();

  // Get normalized (phi) name for a legacy (fluid) op name.
  std::string GetOpName(const std::string& legacy_name) const;

  // Get legacy arg name for a normalized (phi) op/arg name.
  std::string GetLegacyArgName(const std::string& op_type,
                               const std::string& arg_name) const;

 private:
  OpNameNormalizer();

  // op_name_mappings: legacy_name → phi_name
  std::unordered_map<std::string, std::string> op_name_mappings_;
  // op_arg_name_mappings: phi_op_name → {phi_arg_name → legacy_arg_name}
  std::unordered_map<std::string, std::unordered_map<std::string, std::string>>
      op_arg_name_mappings_;
};

// ---------------------------------------------------------------------------
// OpYamlInfoParser — resolve named inputs/outputs to positional indices.
// Data extracted from PaddlePaddle's ops.yaml.
// ---------------------------------------------------------------------------
class OpYamlInfoParser {
 public:
  // Create for a specific op type.
  explicit OpYamlInfoParser(const std::string& op_type);

  // Map named input to positional index.
  int32_t InputNameToIndex(const std::string& name) const;

  // Map named output to positional index.
  int32_t OutputNameToIndex(const std::string& name) const;

  bool HasInput(const std::string& name) const;
  bool HasOutput(const std::string& name) const;

 private:
  std::unordered_map<std::string, int32_t> input_map_;
  std::unordered_map<std::string, int32_t> output_map_;
};

// ---------------------------------------------------------------------------
// Global info — load from built-in data
// ---------------------------------------------------------------------------
// Set the PIR version to use for lookups (e.g., "3.4", "3.3", etc.)
void SetPirVersion(const std::string& version);
std::string GetPirVersion();

}  // namespace pir
}  // namespace paddle2onnx
