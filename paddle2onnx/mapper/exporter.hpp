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
#include <onnx/checker.h>
#include <onnx/onnx_pb.h>
#include <algorithm>
#include <set>

#include "paddle2onnx/mapper/activation.hpp"
#include "paddle2onnx/mapper/elementwise.hpp"
#include "paddle2onnx/mapper/nn.hpp"
#include "paddle2onnx/parser/parser.hpp"

namespace paddle2onnx {

struct ModelExporter {
 private:
  std::vector<std::shared_ptr<ONNX_NAMESPACE::NodeProto>> parameters;
  std::vector<std::shared_ptr<ONNX_NAMESPACE::ValueInfoProto>> inputs;
  std::vector<std::shared_ptr<ONNX_NAMESPACE::ValueInfoProto>> outputs;
  std::vector<std::shared_ptr<ONNX_NAMESPACE::NodeProto>> op_nodes;

  void ExportParameters(const std::map<std::string, Weight>& params,
                        bool use_initializer = false);
  void ExportInputOutputs(const std::vector<TensorInfo>& input_infos,
                          const std::vector<TensorInfo>& output_infos);
  void ExportOp(const PaddleParser& parser, int32_t opset_version,
                int64_t block_id, int64_t op_id);
  // Get a proper opset version in range of [7, 15]
  // Also will check the model is convertable, this will include 2 parts
  //    1. is the op convert function implemented
  //    2. is the op convertable(some cases may not be able to convert)
  // If the model is not convertable, return -1
  int32_t GetMinOpset(const PaddleParser& parser, bool verbose = false);

  bool CheckIfOpSupported(const PaddleParser& parser, std::set<std::string>*);

 public:
  std::shared_ptr<ONNX_NAMESPACE::ModelProto> Run(
      const PaddleParser& parser, int opset_version = 9,
      bool auto_upgrade_opset = true, bool verbose = false);
};

void ModelExporter::ExportParameters(
    const std::map<std::string, Weight>& params, bool use_initializer) {
  for (auto& item : params) {
    // TODO I'm not handling use_initializer now, but some day I will
    auto node = MakeConstant(item.first, item.second);
    parameters.push_back(std::move(node));
  }
}

void ModelExporter::ExportInputOutputs(
    const std::vector<TensorInfo>& input_infos,
    const std::vector<TensorInfo>& output_infos) {
  for (auto& item : input_infos) {
    auto value_info = MakeValueInfo(item);
    inputs.push_back(std::move(value_info));
  }
  for (auto& item : output_infos) {
    auto value_info = MakeValueInfo(item);
    outputs.push_back(std::move(value_info));
  }
}

void ModelExporter::ExportOp(const PaddleParser& parser, int32_t opset_version,
                             int64_t block_id, int64_t op_id) {
  auto op = parser.GetOpDesc(block_id, op_id);
  std::vector<std::shared_ptr<ONNX_NAMESPACE::NodeProto>> nodes;

  auto mapper =
      MapperHelper::Get()->CreateMapper(op.type(), parser, block_id, op_id);
  mapper->Run(&nodes, opset_version);
  delete mapper;
  for (size_t i = 0; i < nodes.size(); ++i) {
    op_nodes.push_back(nodes[i]);
  }
}

std::shared_ptr<ONNX_NAMESPACE::ModelProto> ModelExporter::Run(
    const PaddleParser& parser, int opset_version, bool auto_upgrade_opset,
    bool verbose) {
  Assert(opset_version <= 15 && opset_version >= 7,
         "Paddle2ONNX now only support opset version in range of [7, 15].");
  inputs.clear();
  outputs.clear();
  parameters.clear();
  op_nodes.clear();

  // clear name_counter
  // this use to generate unique name
  // for intermdiate
  // while converting all the op
  MapperHelper::Get()->ClearNameCounter();

  std::set<std::string> unsupported_ops;
  if (!CheckIfOpSupported(parser, &unsupported_ops)) {
    std::cerr << "Oops, there are some operators not supported by Paddle2ONNX "
                 "yet, list as below "
              << std::endl;
    for (auto& item : unsupported_ops) {
      std::cerr << "=====: " << item << std::endl;
    }
    Assert(1 == 0,
           "Due to the unsupported operators, the conversion is aborted.");
  }

  int32_t min_opset = GetMinOpset(parser, verbose);
  if (min_opset < 0) {
    min_opset = GetMinOpset(parser, true);
    Assert(false,
           "Model exporting failed, you can report this problem to "
           "https://github.com/PaddlePaddle/Paddle2ONNX.git.");
  }
  if (!auto_upgrade_opset) {
    if (min_opset > opset_version) {
      std::cerr << "This model cannot export to onnx with opset version = "
                << opset_version
                << ", the opset version for this model should be greater or "
                   "equal than "
                << min_opset << std::endl;
      Assert(false, "Due to opset version, the model exporting is aborted.");
    }
  } else {
    if (min_opset > opset_version) {
      std::cerr << "Opset version has been changed to " << min_opset << " from "
                << opset_version << std::endl;
      opset_version = min_opset;
    }
  }

  ExportParameters(parser.params);
  ExportInputOutputs(parser.inputs, parser.outputs);

  // Only convert blocks 0 now
  // because control flow is not supported yet
  for (auto i = 0; i < parser.NumOfOps(0); ++i) {
    auto op = parser.GetOpDesc(0, i);
    if (op.type() == "feed") {
      continue;
    } else if (op.type() == "fetch") {
      continue;
    }
    ExportOp(parser, opset_version, 0, i);
  }

  // construct a onnx model proto
  auto model = std::make_shared<ONNX_NAMESPACE::ModelProto>();
  // TODO ir version is related to onnx version
  model->set_ir_version(IR_VERSION);
  auto graph = model->mutable_graph();
  graph->set_name("Model from PaddlePaddle.");
  auto opset_id = model->add_opset_import();
  // TODO custom op is not considered
  opset_id->set_domain("");
  opset_id->set_version(ONNX_NAMESPACE::Version_MAX);

  for (auto& item : parameters) {
    *(graph->add_node()) = *(item.get());
  }
  for (auto& item : inputs) {
    *(graph->add_input()) = *(item.get());
  }
  for (auto& item : op_nodes) {
    *(graph->add_node()) = (*item.get());
  }
  for (auto& item : outputs) {
    *(graph->add_output()) = (*item.get());
  }

  // TODO
  // If we need to integrate with framework
  // this check will return a information
  // to let framework know the conversion is
  // pass or fail
  ONNX_NAMESPACE::checker::check_model(*(model.get()));
  if (verbose) {
    std::cerr << "ONNX Model exported successed!" << std::endl;
  }
  return model;
}

bool ModelExporter::CheckIfOpSupported(const PaddleParser& parser,
                                       std::set<std::string>* unsupported_ops) {
  unsupported_ops->clear();
  for (auto i = 0; i < parser.NumOfBlocks(); ++i) {
    for (auto j = 0; j < parser.NumOfOps(i); ++j) {
      auto op = parser.GetOpDesc(i, j);
      if (op.type() == "feed" || op.type() == "fetch") {
        continue;
      }
      if (!MapperHelper::Get()->IsRegistered(op.type())) {
        unsupported_ops->insert(op.type());
      }
    }
  }
  return (unsupported_ops->size() == 0);
}

int32_t ModelExporter::GetMinOpset(const PaddleParser& parser, bool verbose) {
  int32_t max_opset = -1;
  bool exportable = true;
  for (auto i = 0; i < parser.NumOfBlocks(); ++i) {
    for (auto j = 0; j < parser.NumOfOps(i); ++j) {
      auto op = parser.GetOpDesc(i, j);
      if (op.type() == "feed" || op.type() == "fetch") {
        continue;
      }
      auto mapper = MapperHelper::Get()->CreateMapper(op.type(), parser, i, j);
      int32_t current_min_opset = mapper->GetMinOpset(verbose);
      if (current_min_opset < 0) {
        exportable = false;
      } else if (current_min_opset > max_opset) {
        max_opset = current_min_opset;
      }
      delete mapper;
    }
  }

  // Here we put some checks to make sure
  // paddle2onnx could compatible with
  // other version of onnx
  int32_t max_support_opset = MAX_ONNX_OPSET_VERSION;
  if (exportable && (max_opset > MAX_ONNX_OPSET_VERSION)) {
    exportable = false;
    if (verbose) {
      std::cerr << "[ERROR] The compiled onnx version only support opset 7~"
                << MAX_ONNX_OPSET_VERSION
                << ", but now this model need at least opset " << max_opset
                << ", please compile with higher version of onnx." << std::endl;
    }
  }
  if (exportable) {
    return max_opset;
  }

  return -1;
}
}  // namespace paddle2onnx
