// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
#include <onnx/onnx_pb.h>

#include <algorithm>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "paddle2onnx/mapper/mapper.h"
#include "paddle2onnx/mapper/quantize_helper.h"
#include "paddle2onnx/parser/parser.h"
#include "paddle2onnx/parser/pir_parser.h"

#ifdef _MSC_VER
#define PATH_SEP "\\"
#else
#define PATH_SEP "/"
#endif

inline std::string convert_pir_op_name(const std::string pir_op_name) {
  std::unordered_map<std::string, std::string> op_name_mappings = {
      {"matmul", "matmul_v2"},
      // {"relu", "relu6"},
      {"batch_norm_", "batch_norm"},
      {"topk", "top_k_v2"},
      {"repeat_interleave_with_tensor_index", "repeat_interleave"},
      {"expand", "expand_v2"},
      {"assign_value_", "assign_value"},
      {"flatten", "flatten_contiguous_range"},
      {"unsqueeze", "unsqueeze2"},
      {"arange", "range"},
      {"argmax", "arg_max"},
      {"floor_divide", "elementwise_floordiv"},
      {"subtract", "elementwise_sub"},
      {"multiply", "elementwise_mul"},
      {"divide", "elementwise_div"},
      {"remainder", "elementwise_mod"},
      {"minimum", "elementwise_min"},
      {"maximum", "elementwise_max"},
      {"min", "reduce_min"},
      {"max", "reduce_max"},
      {"mean", "reduce_mean"},
      {"sum", "reduce_sum"},
      {"prod", "reduce_prod"},
      {"any", "reduce_any"},
      {"all", "reduce_all"},
      {"numel", "size"},
      {"hardswish", "hard_swish"},
      {"hardsigmoid", "hard_sigmoid"},
      {"add", "elementwise_add"},
      {"add_n", "sum"},
      {"grid_sample", "grid_sampler"},
      {"nonzero", "where_index"},
      {"topk", "top_k"}};
  std::string op_name = pir_op_name;
  std::string prefix = "pd_op.";
  std::string builtin_prefix = "builtin.";

  size_t prefix_pos = op_name.find(prefix);
  if (prefix_pos != std::string::npos) {
    op_name = op_name.substr(prefix_pos + prefix.size());
  } else {
    if (op_name.substr(0, builtin_prefix.size()) == builtin_prefix) {
      op_name[builtin_prefix.size() - 1] = '_';
    }
  }
  auto it = op_name_mappings.find(op_name);
  if (it != op_name_mappings.end()) {
    op_name = it->second;
  }

  return op_name;
}

inline std::string GetFilenameFromPath(const std::string& path) {
  auto pos = path.find_last_of(PATH_SEP);
  if (pos == std::string::npos) {
    return path;
  }
  return path.substr(pos + 1);
}

namespace paddle2onnx {
class ModelExporter {
 public:
  QuantizeModelProcessor quantize_model_processer;

  void SaveExternalData(ONNX_NAMESPACE::GraphProto* graph,
                        const std::string& external_file_path,
                        bool* save_external = nullptr);

  void ONNXChecker(const ONNX_NAMESPACE::ModelProto& model,
                   const bool& verbose);

  std::string Run(const PaddleParser& parser,
                  int opset_version = 9,
                  bool auto_upgrade_opset = true,
                  bool verbose = false,
                  bool enable_onnx_checker = true,
                  bool enable_experimental_op = false,
                  bool enable_optimize = true,
                  const std::string& deploy_backend = "onnxruntime",
                  std::string* calibration_cache = nullptr,
                  const std::string& external_file = "",
                  bool* save_external = nullptr,
                  bool export_fp16_model = false,
                  std::vector<std::string> disable_fp16_op_types = {});
  std::string Run(PaddlePirParser
                    & pir_parser,
                  int opset_version = 9,
                  bool auto_upgrade_opset = true,
                  bool verbose = false,
                  bool enable_onnx_checker = true,
                  bool enable_experimental_op = false,
                  bool enable_optimize = true,
                  const std::string& deploy_backend = "onnxruntime",
                  std::string* calibration_cache = nullptr,
                  const std::string& external_file = "",
                  bool* save_external = nullptr,
                  bool export_fp16_model = false,
                  std::vector<std::string> disable_fp16_op_types = {});
 private:
  bool verbose_ = false;
  // The _deploy_backend will pass to Mapper to influence the conversion
  std::string deploy_backend_ = "onnxruntime";
  std::string* calibration_cache_ = nullptr;
  int32_t opset_version_ = 7;

  void ExportWhile(PaddlePirParser& pir_parser,
                                OnnxHelper* temp_helper,
                                pir::Operation* op);
  bool IsOpsRegistered(const PaddleParser& parser, bool enable_experimental_op);
  bool IsOpsRegistered(const PaddlePirParser& parser,
                       bool enable_experimental_op);

  ONNX_NAMESPACE::ModelProto onnx_model_;
  // Opset Version
  int32_t GetMinOpsetVersion(const PaddleParser& parser);
  int32_t GetMinOpsetVersion(const PaddlePirParser& parser);
  void SetOpsetVersion(const PaddleParser& parser, bool auto_upgrade_opset);
  void SetOpsetVersion(const PaddlePirParser& pir_parser,
                       bool auto_upgrade_opset);
  // IR Version
  inline ONNX_NAMESPACE::Version GetIRVersion() const;
  void SetIRVersion();
  //
  void ExportInputOutputs(
      const PaddleParser& parser,
      std::vector<std::shared_ptr<ONNX_NAMESPACE::ValueInfoProto>>
        & inputs,
      std::vector<std::shared_ptr<ONNX_NAMESPACE::ValueInfoProto>>
        & outputs);

  void ExportInputOutputs(
      const PaddlePirParser& pir_parser,
      std::vector<std::shared_ptr<ONNX_NAMESPACE::ValueInfoProto>>
        & inputs,
      std::vector<std::shared_ptr<ONNX_NAMESPACE::ValueInfoProto>>
        & outputs);

  void ExportParameters(
      const PaddleParser& parser,
      std::vector<std::shared_ptr<ONNX_NAMESPACE::NodeProto>>
        & parameters);
  void ExportParameters(
      const PaddlePirParser& pir_parser,
      std::vector<std::shared_ptr<ONNX_NAMESPACE::NodeProto>>
        & parameters);
  // Process dumplicate tensor names in paddle model
  std::set<std::string> tensor_names_;
  void ProcessGraphDumplicateNames(
      std::vector<std::shared_ptr<ONNX_NAMESPACE::NodeProto>>
        & parameters,
      std::vector<std::shared_ptr<ONNX_NAMESPACE::ValueInfoProto>>
        & inputs,
      std::vector<std::shared_ptr<ONNX_NAMESPACE::ValueInfoProto>>
        & outputs,
      std::vector<std::shared_ptr<ONNX_NAMESPACE::NodeProto>>
        & nodes,
      std::map<std::string, QuantizeInfo>
        & quantize_info);
  // Update constant node in parameters. When process quantize model, the
  // weight dtype may be int8, it should be convet to float32 and use this
  // function to update converted params.
  void UpdateParameters(
      const std::map<std::string, Weight>& params,
      std::vector<std::shared_ptr<ONNX_NAMESPACE::NodeProto>>
        & parameters);
  //
  std::map<std::string, std::pair<int32_t, int32_t>> sub_block_map_;
  ONNX_NAMESPACE::GraphProto ExportConditionalBlock(
      const PaddleParser& parser,
      int32_t block_id,
      int32_t op_id,
      const std::string& output_names);

  ONNX_NAMESPACE::GraphProto ExportIfBlock(PaddlePirParser
                                            & pir_parser,
                                           pir::Block
                                            & block);

  ONNX_NAMESPACE::GraphProto ExportBlock(
      const PaddleParser& parser,
      int32_t block_id,
      std::vector<std::shared_ptr<ONNX_NAMESPACE::NodeProto>>
        & parameters,
      std::vector<std::shared_ptr<ONNX_NAMESPACE::ValueInfoProto>>
        & inputs,
      std::vector<std::shared_ptr<ONNX_NAMESPACE::ValueInfoProto>>
        & outputs);
  ONNX_NAMESPACE::GraphProto ExportBlock(
      PaddlePirParser
        & pir_parser,
      pir::Block* block,
      std::vector<std::shared_ptr<ONNX_NAMESPACE::NodeProto>>& parameters,
      std::vector<std::shared_ptr<ONNX_NAMESPACE::ValueInfoProto>>& inputs,
      std::vector<std::shared_ptr<ONNX_NAMESPACE::ValueInfoProto>>& outputs,
      bool if_in_subblock,bool is_while_block);

  void ExportOp(const PaddleParser& parser,
                OnnxHelper* helper,
                int32_t opset_version,
                int64_t block_id,
                int64_t op_id,
                bool verbose);
  void ExportOp(const PaddlePirParser& pir_parser,
                OnnxHelper* helper,
                int32_t opset_version,
                pir::Operation* op,
                int64_t op_id,
                bool if_in_subblock,
                bool verbose);
#if 0
    bool IsLoopSupported(const PaddleParser &parser, const int64_t &block_id,
                         const int64_t &op_id);
    void ExportLoop(const PaddleParser &parser, OnnxHelper *helper,
                    int32_t opset_version, int64_t block_id, int64_t op_id,
                    bool verbose);
#endif
  ONNX_NAMESPACE::ModelProto Optimize(const ONNX_NAMESPACE::ModelProto& model);
};
}  // namespace paddle2onnx
