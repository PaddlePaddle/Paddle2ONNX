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
#include <vector>

namespace paddle2onnx {
namespace pir {

// ===========================================================================
// Known versions — order matters: latest first for best match
// ===========================================================================
const std::vector<std::string>& OpYamlInfoParser::KnownVersions() {
  static const std::vector<std::string> versions = {
      "v3_4", "v3_3", "v3_2", "v3_1", "v3_0",
  };
  return versions;
}

// ===========================================================================
// OpNameNormalizer — combined from ALL versions
// ===========================================================================
OpNameNormalizer* OpNameNormalizer::Instance() {
  static OpNameNormalizer instance;
  return &instance;
}

OpNameNormalizer::OpNameNormalizer() {
  // Combined table (v3_4) already contains ops from v3.0~v3.4
  op_name_mappings_ = GetOpNameMappings_combined();
}

std::string OpNameNormalizer::GetOpName(const std::string& legacy_name) const {
  auto it = op_name_mappings_.find(legacy_name);
  return it != op_name_mappings_.end() ? it->second : legacy_name;
}

std::string OpNameNormalizer::GetLegacyArgName(
    const std::string& op_type, const std::string& arg_name) const {
  auto& arg_map = GetOpArgMappings_combined(op_type);
  auto it = arg_map.find(arg_name);
  return it != arg_map.end() ? it->second : arg_name;
}

std::string OpNameNormalizer::GetDirectMapping(
    const std::string& op_type, const std::string& arg_name) const {
  return GetLegacyArgName(op_type, arg_name);
}

// ===========================================================================
// OpYamlInfoParser — auto-detect version
// ===========================================================================
OpYamlInfoParser::OpYamlInfoParser(const std::string& op_type) {
  // Try combined table first (generated from v3.0~v3.4, contains all ops)
  const auto& combined_input = GetOpInputIndices_combined(op_type);
  if (!combined_input.empty()) {
    for (auto& pair : combined_input)
      input_map_[pair.first] = static_cast<int32_t>(pair.second);
    for (auto& pair : GetOpOutputIndices_combined(op_type))
      output_map_[pair.first] = static_cast<int32_t>(pair.second);
    return;
  }

  // Fallback: try each version individually (for future compatibility
  // when version-specific op definitions diverge)
  for (auto& ver : KnownVersions()) {
    // TryVersion dispatches to version-specific table
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
