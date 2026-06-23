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
// Op name compatibility and info lookup — uses auto-generated tables
// from ops.yaml + op_compat.yaml. Version auto-detection: when looking
// up an op, each version table is tried in order (latest first) until
// a match is found.

#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "paddle2onnx/pir/pir_op_info_generated.h"

namespace paddle2onnx {
namespace pir {

// ---------------------------------------------------------------------------
// OpNameNormalizer — combines mappings from ALL known versions.
// ---------------------------------------------------------------------------
class OpNameNormalizer {
 public:
  static OpNameNormalizer* Instance();

  std::string GetOpName(const std::string& legacy_name) const;
  std::string GetLegacyArgName(const std::string& op_type,
                               const std::string& arg_name) const;
  std::string GetDirectMapping(const std::string& op_type,
                               const std::string& arg_name) const;

 private:
  OpNameNormalizer();
  std::unordered_map<std::string, std::string> op_name_mappings_;
  std::unordered_map<std::string,
                     std::unordered_map<std::string, std::string>>
      op_arg_name_mappings_;
};

// ---------------------------------------------------------------------------
// OpYamlInfoParser — resolve named inputs/outputs to positional indices.
// Auto-detects version: tries each known version table in order (latest
// first) and uses the first one that has the op.
// ---------------------------------------------------------------------------
class OpYamlInfoParser {
 public:
  explicit OpYamlInfoParser(const std::string& op_type);

  int32_t InputNameToIndex(const std::string& name) const;
  int32_t OutputNameToIndex(const std::string& name) const;
  bool HasInput(const std::string& name) const;
  bool HasOutput(const std::string& name) const;

 private:
  std::unordered_map<std::string, int32_t> input_map_;
  std::unordered_map<std::string, int32_t> output_map_;

  // Known versions in order (latest first → best match)
  static const std::vector<std::string>& KnownVersions();
  // Try to initialize from a specific version; returns true if op found
  bool TryVersion(const std::string& version, const std::string& op_type);
};

}  // namespace pir
}  // namespace paddle2onnx
