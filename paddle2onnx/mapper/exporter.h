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
#include <set>

#include "paddle2onnx/mapper/mapper.h"
#include "paddle2onnx/mapper/quantize_helper.h"
#include "paddle2onnx/parser/parser.h"

#ifdef _MSC_VER
#define PATH_SEP "\\"
#else
#define PATH_SEP "/"
#endif

inline std::string GetFilenameFromPath(const std::string &path)
{
  auto pos = path.find_last_of(PATH_SEP);
  if (pos == std::string::npos)
  {
    return path;
  }
  return path.substr(pos + 1);
}

namespace paddle2onnx
{
  struct ModelExporter
  {
  public:
    QuantizeModelProcessor quantize_model_processer;

    //  // Remove isolated nodes in onnx model
    //  void RemoveIsolatedNodes(
    //      std::vector<std::shared_ptr<ONNX_NAMESPACE::NodeProto>>* parameters,
    //      std::vector<std::shared_ptr<ONNX_NAMESPACE::ValueInfoProto>>* inputs,
    //      std::vector<std::shared_ptr<ONNX_NAMESPACE::ValueInfoProto>>* outputs,
    //      std::vector<std::shared_ptr<ONNX_NAMESPACE::NodeProto>>* nodes);
    // Process dumplicate tensor names in paddle model
    void ProcessGraphDumplicateNames(
        std::vector<std::shared_ptr<ONNX_NAMESPACE::NodeProto>> *parameters,
        std::vector<std::shared_ptr<ONNX_NAMESPACE::ValueInfoProto>> *inputs,
        std::vector<std::shared_ptr<ONNX_NAMESPACE::ValueInfoProto>> *outputs,
        std::vector<std::shared_ptr<ONNX_NAMESPACE::NodeProto>> *nodes,
        std::map<std::string, QuantizeInfo> *quantize_info = nullptr);

    void SaveExternalData(ONNX_NAMESPACE::GraphProto *graph,
                          const std::string &external_file_path,
                          bool *save_external = nullptr);

    void ONNXChecker(const ONNX_NAMESPACE::ModelProto &model,
                     const bool &verbose);

    std::string Run(const PaddleParser &parser,
                    int opset_version = 9,
                    bool auto_upgrade_opset = true,
                    bool verbose = false,
                    bool enable_onnx_checker = true,
                    bool enable_experimental_op = false,
                    bool enable_optimize = true,
                    const std::string &deploy_backend = "onnxruntime",
                    std::string *calibration_cache = nullptr,
                    const std::string &external_file = "",
                    bool *save_external = nullptr,
                    bool export_fp16_model = false,
                    std::vector<std::string> disable_fp16_op_types = {});

  private:
    bool verbose_ = false;
    // The _deploy_backend will pass to Mapper to influence the conversion
    std::string deploy_backend_ = "onnxruntime";
    int32_t opset_version_ = 7;

    bool IsOpsRegistered(const PaddleParser &parser,
                         bool enable_experimental_op);

    ONNX_NAMESPACE::ModelProto onnx_model_;
    // Opset Version
    int32_t GetMinOpsetVersion(const PaddleParser &parser);
    void SetOpsetVersion(const PaddleParser &parser, bool auto_upgrade_opset);
    // IR Version
    inline ONNX_NAMESPACE::Version GetIRVersion() const;
    void SetIRVersion();
    //
    std::vector<std::shared_ptr<ONNX_NAMESPACE::ValueInfoProto>> inputs;
    std::vector<std::shared_ptr<ONNX_NAMESPACE::ValueInfoProto>> outputs;
    void ExportInputOutputs(const PaddleParser &parser);
    //
    std::vector<std::shared_ptr<ONNX_NAMESPACE::NodeProto>> parameters;
    void ExportParameters(const PaddleParser &parser);

    // Update constant node in parameters. When process quantize model, the weight
    // dtype may be int8, it should be convet to float32 and use this function to
    // update converted params.
    void UpdateParameters(const std::map<std::string, Weight> &params);
    void ExportOp(const PaddleParser &parser,
                  OnnxHelper *helper,
                  int32_t opset_version,
                  int64_t block_id,
                  int64_t op_id,
                  bool verbose);
#if 0
    bool IsLoopSupported(const PaddleParser &parser, const int64_t &block_id,
                         const int64_t &op_id);
    void ExportLoop(const PaddleParser &parser, OnnxHelper *helper,
                    int32_t opset_version, int64_t block_id, int64_t op_id,
                    bool verbose);
#endif
    ONNX_NAMESPACE::ModelProto Optimize(const ONNX_NAMESPACE::ModelProto &model);
  };
} // namespace paddle2onnx
