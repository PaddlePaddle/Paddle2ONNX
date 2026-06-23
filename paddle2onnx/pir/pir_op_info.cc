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

#include "paddle2onnx/pir/pir_op_info.h"

#include <string>

namespace paddle2onnx {
namespace pir {

static std::string g_pir_version = "v3.4";

void SetPirVersion(const std::string& version) { g_pir_version = version; }
std::string GetPirVersion() { return g_pir_version; }

// ===========================================================================
// OpNameNormalizer
// ===========================================================================
OpNameNormalizer* OpNameNormalizer::Instance() {
  static OpNameNormalizer instance;
  return &instance;
}

OpNameNormalizer::OpNameNormalizer() { Initialize(g_pir_version); }

void OpNameNormalizer::Initialize(const std::string& version) {
  // Try each known version; fall back to v3.4
  // Use the generated mapping tables
  op_name_mappings_ = GetOpNameMappings_v3_4();
}

std::string OpNameNormalizer::GetOpName(const std::string& legacy_name) const {
  auto it = op_name_mappings_.find(legacy_name);
  if (it != op_name_mappings_.end()) {
    return it->second;
  }
  return legacy_name;  // passthrough if not found
}

std::string OpNameNormalizer::GetLegacyArgName(
    const std::string& op_type, const std::string& arg_name) const {
  auto& arg_map = GetOpArgMappings_v3_4(op_type);
  auto it = arg_map.find(arg_name);
  if (it != arg_map.end()) {
    return it->second;
  }
  return arg_name;  // passthrough
}

std::string OpNameNormalizer::GetDirectMapping(
    const std::string& op_type, const std::string& arg_name) const {
  return GetLegacyArgName(op_type, arg_name);
}

// ===========================================================================
// OpYamlInfoParser
// ===========================================================================
OpYamlInfoParser::OpYamlInfoParser(const std::string& op_type) {
  auto& input_indices = GetOpInputIndices_v3_4(op_type);
  for (auto& pair : input_indices) {
    input_map_[pair.first] = static_cast<int32_t>(pair.second);
  }

  auto& output_indices = GetOpOutputIndices_v3_4(op_type);
  for (auto& pair : output_indices) {
    output_map_[pair.first] = static_cast<int32_t>(pair.second);
  }
}

int32_t OpYamlInfoParser::InputNameToIndex(const std::string& name) const {
  auto it = input_map_.find(name);
  return it != input_map_.end() ? it->second : -1;
}

int32_t OpYamlInfoParser::OutputNameToIndex(const std::string& name) const {
  auto it = output_map_.find(name);
  return it != output_map_.end() ? it->second : -1;
}

bool OpYamlInfoParser::HasInput(const std::string& name) const {
  return input_map_.find(name) != input_map_.end();
}

bool OpYamlInfoParser::HasOutput(const std::string& name) const {
  return output_map_.find(name) != output_map_.end();
}

}  // namespace pir
}  // namespace paddle2onnx
