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

#include "paddle2onnx/mapper/exporter.h"
#include <onnx/checker.h>
#include <onnx/shape_inference/implementation.h>
#include "onnxoptimizer/optimize.h"

namespace paddle2onnx {
MapperHelper* MapperHelper::helper = nullptr;

void ModelExporter::ExportParameters(
    const std::map<std::string, Weight>& params, bool use_initializer) {
  for (auto& item : params) {
    // TODO(jiangjiajun) I'm not handling use_initializer now, but some day I
    // will
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
  mapper->Run(&helper, opset_version);
  delete mapper;
}

std::string ModelExporter::Run(const PaddleParser& parser, int opset_version,
                               bool auto_upgrade_opset, bool verbose,
                               bool enable_onnx_checker,
                               bool enable_experimental_op,
                               bool enable_optimize) {
  Assert(opset_version <= 15 && opset_version >= 7,
         "Paddle2ONNX now only support opset version in range of [7, 15].");
  helper.Clear();
  inputs.clear();
  outputs.clear();
  parameters.clear();

  // clear name_counter
  // this use to generate unique name
  // for intermdiate
  // while converting all the op
  MapperHelper::Get()->ClearNameCounter();

  std::set<std::string> unsupported_ops;
  if (!CheckIfOpSupported(parser, &unsupported_ops, enable_experimental_op)) {
    std::cerr << "Oops, there are some operators not supported by Paddle2ONNX "
                 "yet, list as below "
              << std::endl;
    for (auto& item : unsupported_ops) {
      std::cerr << "=====: " << item << std::endl;
    }
    Assert(1 == 0,
           "Due to the unsupported operators, the conversion is aborted.");
  }

  int32_t min_opset = GetMinOpset(parser, false);
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
  helper.SetOpsetVersion(opset_version);
  std::cerr << "Model will exported with opset = " << helper.opset_version
            << std::endl;

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
  // TODO(jiangjiajun) ir version is related to onnx version
  model->set_ir_version(ONNX_NAMESPACE::IR_VERSION);
  auto graph = model->mutable_graph();
  graph->set_name("Model from PaddlePaddle.");
  auto opset_id = model->add_opset_import();
  // TODO(jiangjiajun) custom op is not considered
  opset_id->set_domain("");
  opset_id->set_version(opset_version);

  for (auto& item : parameters) {
    *(graph->add_node()) = *(item.get());
  }
  for (auto& item : inputs) {
    *(graph->add_input()) = *(item.get());
  }
  for (auto& item : helper.nodes) {
    *(graph->add_node()) = (*item.get());
  }
  for (auto& item : outputs) {
    *(graph->add_output()) = (*item.get());
  }

  // TODO(jiangjiajun)
  // If we need to integrate with framework
  // this check will return a information
  // to let framework know the conversion is
  // pass or fail
  if (enable_onnx_checker) {
    ONNX_NAMESPACE::checker::check_model(*(model.get()));
    std::cerr << "[Paddle2ONNX] ONNX model conversion is valid." << std::endl;
    ONNX_NAMESPACE::shape_inference::InferShapes(*(model.get()));
    std::cerr << "[Paddle2ONNX] Shape Inference done with ONNX model."
              << std::endl;
  }

  std::string out;
  if (enable_optimize) {
    auto const opt_model = Optimize(*(model.get()));
    if (!opt_model.SerializeToString(&out)) {
      if (verbose) {
        std::cerr << "ONNX Model SerializeToString error" << std::endl;
      }
      return "";
    }
  } else {
    if (!model->SerializeToString(&out)) {
      if (verbose) {
        std::cerr << "ONNX Model SerializeToString error" << std::endl;
      }
      return "";
    }
  }
  return out;
}

bool ModelExporter::CheckIfOpSupported(const PaddleParser& parser,
                                       std::set<std::string>* unsupported_ops,
                                       bool enable_experimental_op) {
  unsupported_ops->clear();
  for (auto i = 0; i < parser.NumOfBlocks(); ++i) {
    for (auto j = 0; j < parser.NumOfOps(i); ++j) {
      auto op = parser.GetOpDesc(i, j);
      if (op.type() == "feed" || op.type() == "fetch") {
        continue;
      }
      if (!MapperHelper::Get()->IsRegistered(op.type())) {
        unsupported_ops->insert(op.type());
      } else if (!enable_experimental_op) {
        auto mapper =
            MapperHelper::Get()->CreateMapper(op.type(), parser, i, j);
        if (mapper->IsExperimentalOp()) {
          unsupported_ops->insert(op.type());
        }
        delete mapper;
      }
    }
  }
  return (unsupported_ops->size() == 0);
}

int32_t ModelExporter::GetMinOpset(const PaddleParser& parser, bool verbose) {
  int32_t max_opset = -1;
  bool exportable = true;
  // Record the number of ops that need to be converted
  int converted_op_num = 0;
  for (auto i = 0; i < parser.NumOfBlocks(); ++i) {
    for (auto j = 0; j < parser.NumOfOps(i); ++j) {
      auto op = parser.GetOpDesc(i, j);
      if (op.type() == "feed" || op.type() == "fetch") {
        continue;
      }
      converted_op_num += 1;
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
  // If there are only feed and fetch op in Paddle model
  if (!converted_op_num) {
    return 7;
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

ONNX_NAMESPACE::ModelProto ModelExporter::Optimize(
    const ONNX_NAMESPACE::ModelProto& model) {
  std::vector<std::string> passes =
      ONNX_NAMESPACE::optimization::GetFuseAndEliminationPass();
  return ONNX_NAMESPACE::optimization::Optimize(model, passes);
}

}  // namespace paddle2onnx
