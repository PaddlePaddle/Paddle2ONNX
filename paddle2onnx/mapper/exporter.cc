// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include <google/protobuf/message.h>
#include <onnx/checker.h>

#include <array>

#include "onnxoptimizer/optimize.h"
#include "paddle/fluid/pir/dialect/operator/ir/control_flow_op.h"
#include "paddle/phi/core/enforce.h"
#include "paddle2onnx/mapper/quantize/ort_quantize_processor.h"
#include "paddle2onnx/mapper/quantize/other_quantize_processor.h"
#include "paddle2onnx/mapper/quantize/rknn_quantize_processor.h"
#include "paddle2onnx/mapper/quantize/tensorrt_quantize_processor.h"
#include "paddle2onnx/optimizer/convert_fp32_to_fp16.h"
#include "paddle2onnx/optimizer/eliminate_non_transpose.h"
#include "paddle2onnx/optimizer/fuse_constant_cast.h"
#include "paddle2onnx/optimizer/fuse_constant_reshape.h"
#include "paddle2onnx/optimizer/fuse_constant_unsqueeze.h"
#include "paddle2onnx/optimizer/fuse_paddle_conv_bias.h"
#include "paddle2onnx/optimizer/fuse_unsqueeze_conv2d_squeeze.h"

namespace paddle2onnx {
MapperHelper* MapperHelper::helper = nullptr;
int32_t OnnxHelper::opset_version = 7;
bool ModelExporter::IsOpsRegistered(const PaddlePirParser& pir_parser,
                                    bool enable_experimental_op) {
  OnnxHelper temp_helper;
  std::set<std::string> unsupported_ops;
  for (auto op : pir_parser.total_blocks_ops) {
    if (op->name() == "pd_op.data" || op->name() == "pd_op.feed" ||
        op->name() == "pd_op.fetch") {
      continue;
    }
    if (op->name() == "pd_op.if" || op->name() == "pd_op.while" ||
        op->name() == "cf.yield") {
      continue;
    }
    if (op->name() == "pd_op.print") {
      continue;
    }
    std::string op_name = convert_pir_op_name(op->name());
    if (!MapperHelper::Get()->IsRegisteredInPir(op_name)) {
      unsupported_ops.insert(op_name);
    }
  }
  // TODO(wangmingkai02) : judge op whether is experimental op
  if (unsupported_ops.size() != 0) {
    auto logger = P2OLogger();
    logger << "There are some ops not supported yet, including ";
    for (auto& item : unsupported_ops) {
      logger << item << ",";
    }
    logger << std::endl;
  }
  return (unsupported_ops.size() == 0);
}

bool ModelExporter::IsWhileSupported(const PaddleParser& parser,
                                     const int64_t& block_id,
                                     const int64_t& op_id) {
  auto x_info = parser.GetOpInput(block_id, op_id, "X");
  auto out_info = parser.GetOpOutput(block_id, op_id, "Out");
  auto cond_info = parser.GetOpInput(block_id, op_id, "Condition");
  std::set<std::string> input_names;
  for (size_t i = 0; i < x_info.size(); ++i) {
    input_names.insert(x_info[i].name);
  }
  input_names.insert(cond_info[0].name);

  for (size_t i = 0; i < out_info.size(); ++i) {
    auto iter = input_names.find(out_info[i].name);
    if (iter == input_names.end()) {
      P2OLogger() << "Cannot find output:" << out_info[i].name
                  << " in input tensors while converting operator 'while', "
                     "Paddle2ONNX doesn't support this situation now."
                  << std::endl;
      return false;
    }
  }

  for (size_t i = 0; i < x_info.size(); ++i) {
    if (x_info[i].is_tensor_array) {
      P2OLogger() << "DenseTensorArray is not supported." << std::endl;
      return false;
    }
  }
  return true;
}

bool ModelExporter::IsOpsRegistered(const PaddleParser& parser,
                                    bool enable_experimental_op) {
  OnnxHelper temp_helper;
  std::set<std::string> unsupported_ops;
  for (auto i = 0; i < parser.NumOfBlocks(); ++i) {
    for (auto j = 0; j < parser.NumOfOps(i); ++j) {
      auto op = parser.GetOpDesc(i, j);
      if (op.type() == "feed" || op.type() == "fetch") {
        continue;
      } else if (op.type() == "conditional_block" ||
                 op.type() == "select_input") {
        continue;
      } else if (op.type() == "while" && enable_experimental_op) {
        if (!IsWhileSupported(parser, i, j)) {
          unsupported_ops.insert("while");
        }
        continue;
      }

      if (custom_ops.find(op.type()) != custom_ops.end()) {
        continue;
      }
      if (!MapperHelper::Get()->IsRegistered(op.type())) {
        unsupported_ops.insert(op.type());
      } else if (!enable_experimental_op) {
        auto mapper = MapperHelper::Get()->CreateMapper(
            op.type(), parser, &temp_helper, i, j);
        if (mapper->IsExperimentalOp()) {
          unsupported_ops.insert(op.type());
        }
        delete mapper;
      }
    }
  }
  if (unsupported_ops.size() == 0) {
    return true;
  }

  auto logger = P2OLogger();
  logger << "Oops, there are some operators not supported yet, including ";
  for (auto& item : unsupported_ops) {
    logger << item << ",";
  }
  logger << std::endl;
  return false;
}

int32_t ModelExporter::GetMinOpsetVersion(const PaddleParser& parser) {
  int32_t max_opset = 7;
  std::set<std::string> verbose_log;
  OnnxHelper helper;
  for (auto i = 0; i < parser.NumOfBlocks(); ++i) {
    for (auto j = 0; j < parser.NumOfOps(i); ++j) {
      auto op = parser.GetOpDesc(i, j);
      if (custom_ops.find(op.type()) != custom_ops.end()) {
        continue;
      }

      // Skip the input and output nodes.
      if (op.type() == "feed" || op.type() == "fetch" ||
          op.type() == "conditional_block") {
        continue;
      }

      int current_opset = 7;
      if (op.type() == "select_input") {
        P2OLogger() << "Detected there's control flow "
                       "op('conditional_block/select_input') in your model, "
                    << "this requires the minimal opset version of 11."
                    << std::endl;
        current_opset = 11;
      } else if (op.type() == "while") {
        P2OLogger()
            << "Detected there's control flow 'while' op in your model, "
            << "this requires the minimal opset version of 13." << std::endl;
        current_opset = 13;
      } else {
        auto mapper =
            MapperHelper::Get()->CreateMapper(op.type(), parser, &helper, i, j);
        current_opset = mapper->GetMinOpsetVersion(verbose_);
        delete mapper;
      }

      if (current_opset > max_opset) {
        max_opset = current_opset;
        if (current_opset > opset_version_) {
          verbose_log.insert("Due to the operator: " + op.type() + ", " +
                             "requires opset_version >= " +
                             std::to_string(current_opset) + ".");
        }
      }
    }
  }

  for (auto iter = verbose_log.begin(); iter != verbose_log.end(); ++iter) {
    P2OLogger(verbose_) << *iter << std::endl;
  }
  return max_opset;
}

int32_t ModelExporter::GetCfBlockMinOpsetVersion(
    const PaddlePirParser& pir_parser, pir::Block& block) {
  std::vector<pir::Operation*> sub_blocks_ops_copy(pir_parser.sub_blocks_ops);
  pir_parser.sub_blocks_ops.clear();
  std::vector<pir::Operation*> block_ops;
  for (auto& op : block.ops()) {
    if (op->name() != "builtin.parameter") {
      pir_parser.sub_blocks_ops.push_back(op);
    }
  }
  // Must generate All sub_block's op output names must be generated here
  // because it's may used in OPMapper.GetMinOpsetVersion function.
  pir_parser.GetAllSubBlockOpOutputName(pir_parser.sub_blocks_ops);
  auto max_opset = GetMinOpsetVersion(pir_parser, &block, true);
  pir_parser.sub_blocks_ops.clear();
  pir_parser.sub_blocks_ops = sub_blocks_ops_copy;
  return max_opset;
}

int32_t ModelExporter::GetMinOpsetVersion(const PaddlePirParser& pir_parser,
                                          pir::Block* block,
                                          bool if_in_sublock) {
  int32_t max_opset = 7;
  std::set<std::string> verbose_log;
  OnnxHelper helper;
  std::vector<pir::Operation*> block_ops;
  // it's  necessary to be same with global/sub_blocks_ops
  for (auto& op : block->ops()) {
    if (op->name() != "builtin.parameter") {
      block_ops.push_back(op);
    }
  }
  for (auto i = 0; i < block_ops.size(); ++i) {
    auto op = block_ops[i];
    std::string op_name = op->name();
    if (op_name == "pd_op.data" || op_name == "pd_op.feed" ||
        op_name == "pd_op.fetch" || op_name == "cf.yield" ||
        op_name == "pd_op.print") {
      continue;
    }
    int current_opset = 7;
    if (op_name == "pd_op.if") {
      auto if_op = op->dyn_cast<paddle::dialect::IfOp>();
      pir::Block& true_block = if_op.true_block();
      auto true_block_opset_version =
          GetCfBlockMinOpsetVersion(pir_parser, true_block);
      pir::Block& false_block = if_op.false_block();
      auto false_block_opset_version =
          GetCfBlockMinOpsetVersion(pir_parser, false_block);
      current_opset = true_block_opset_version > false_block_opset_version
                          ? true_block_opset_version
                          : false_block_opset_version;
      current_opset = current_opset > 11 ? current_opset : 11;
    } else if (op_name == "pd_op.while") {
      auto while_op = op->dyn_cast<paddle::dialect::WhileOp>();
      pir_parser.GetWhileInputValuesAndArgsMappings(&while_op);
      current_opset = GetCfBlockMinOpsetVersion(pir_parser, while_op.body());
      current_opset = current_opset > 11 ? current_opset : 11;

    } else {
      auto mapper = MapperHelper::Get()->CreateMapper(
          convert_pir_op_name(op_name), pir_parser, &helper, i, if_in_sublock);
      current_opset = mapper->GetMinOpsetVersion(verbose_);
      delete mapper;
    }
    if (current_opset > max_opset) {
      max_opset = current_opset;
      if (current_opset > opset_version_) {
        if (opset_version_ < 11 ||
            (op_name != "pd_op.if" && op_name != "pd_op.while")) {
          verbose_log.insert("Due to the operator: " + op_name + " " +
                             "requires opset_version >= " +
                             std::to_string(current_opset) + ".");
        }
      }
    }
  }

  for (auto iter = verbose_log.begin(); iter != verbose_log.end(); ++iter) {
    P2OLogger(verbose_) << *iter << std::endl;
  }
  return max_opset;
}

void ModelExporter::SetOpsetVersion(const PaddlePirParser& pir_parser,
                                    bool auto_upgrade_opset) {
  bool opset_is_legal = true;
  // here
  int32_t min_opset =
      GetMinOpsetVersion(pir_parser, pir_parser.pir_program_->block(), false);
  if (min_opset < 7 || min_opset > MAX_ONNX_OPSET_VERSION) {
    P2OLogger(verbose_) << "The Opset Version must be between 7 and "
                        << MAX_ONNX_OPSET_VERSION << std::endl;
    opset_is_legal = false;
  }
  if (!auto_upgrade_opset) {
    if (min_opset > opset_version_) {
      P2OLogger(verbose_) << "Please set the opset_version to "
                          << std::to_string(min_opset)
                          << " or set auto_upgrade_opset=true." << std::endl;
      opset_is_legal = false;
    }
  } else {
    if (min_opset > opset_version_) {
      P2OLogger(verbose_) << "Opset version will change to " << min_opset
                          << " from " << opset_version_ << std::endl;
      opset_version_ = min_opset;
    }
  }
  Assert(opset_is_legal,
         "Due to opset version, the model exporting is aborted.");

  OnnxHelper::SetOpsetVersion(opset_version_);

  auto opset_import = onnx_model_.add_opset_import();
  opset_import->set_domain("");
  opset_import->set_version(opset_version_);
  P2OLogger(verbose_) << "Use opset_version = " << opset_version_
                      << " for ONNX export." << std::endl;
}

void ModelExporter::SetOpsetVersion(const PaddleParser& parser,
                                    bool auto_upgrade_opset) {
  // Set the Opset Version of the ONNX model.
  bool opset_is_legal = true;
  int32_t min_opset = GetMinOpsetVersion(parser);
  if (min_opset < 7 || min_opset >= MAX_ONNX_OPSET_VERSION) {
    P2OLogger(verbose_) << "The Opset Version must be between 7 and "
                        << MAX_ONNX_OPSET_VERSION - 1 << std::endl;
    opset_is_legal = false;
  }
  if (!auto_upgrade_opset) {
    if (min_opset > opset_version_) {
      P2OLogger(verbose_) << "Please set the opset_version to "
                          << std::to_string(opset_version_)
                          << " or set auto_upgrade_opset=true." << std::endl;
      opset_is_legal = false;
    }
  } else {
    if (min_opset > opset_version_) {
      P2OLogger(verbose_) << "Opset version will change to " << min_opset
                          << " from " << opset_version_ << std::endl;
      opset_version_ = min_opset;
    }
  }
  Assert(opset_is_legal,
         "Due to opset version, the model exporting is aborted.");

  OnnxHelper::SetOpsetVersion(opset_version_);

  auto opset_import = onnx_model_.add_opset_import();
  opset_import->set_domain("");
  opset_import->set_version(opset_version_);
  P2OLogger(verbose_) << "Use opset_version = " << opset_version_
                      << " for ONNX export." << std::endl;
  if (custom_ops.size()) {
    auto opset_paddle_id = onnx_model_.add_opset_import();
    opset_paddle_id->set_domain("Paddle");
    opset_paddle_id->set_version(1);
  }
}

inline ONNX_NAMESPACE::Version ModelExporter::GetIRVersion() const {
  int ir_version = 0;
  switch (opset_version_) {
    case 7:
    case 8:
      ir_version = 3;
      break;
    case 9:
      ir_version = 4;
      break;
    case 10:
      ir_version = 5;
      break;
    case 11:
      ir_version = 6;
      break;
    case 12:
    case 13:
    case 14:
      ir_version = 7;
      break;
    case 15:
    case 16:
    case 17:
    case 18:
      ir_version = 8;
      break;
    case 19:
    case 20:
      ir_version = 9;
      break;
    case 21:
      ir_version = 10;
      break;
    default:
      P2OLogger(verbose_) << "The Opset Version must be between 7 and 21."
                          << std::endl;
      Assert(false, "Due to opset version, the model exporting is aborted.");
  }
  return static_cast<ONNX_NAMESPACE::Version>(ir_version);
}

void ModelExporter::SetIRVersion() {
  onnx_model_.set_ir_version(GetIRVersion());
}

void ModelExporter::ExportInputOutputs(
    const PaddleParser& parser,
    std::vector<std::shared_ptr<ONNX_NAMESPACE::ValueInfoProto>>& inputs,
    std::vector<std::shared_ptr<ONNX_NAMESPACE::ValueInfoProto>>& outputs) {
  inputs.clear();
  for (auto& item : parser.inputs) {
    auto value_info = MakeValueInfo(item);
    inputs.push_back(std::move(value_info));
  }
  outputs.clear();
  for (auto& item : parser.outputs) {
    auto value_info = MakeValueInfo(item);
    outputs.push_back(std::move(value_info));
  }
}

void ModelExporter::ExportInputOutputs(
    const PaddlePirParser& pir_parser,
    std::vector<std::shared_ptr<ONNX_NAMESPACE::ValueInfoProto>>& inputs,
    std::vector<std::shared_ptr<ONNX_NAMESPACE::ValueInfoProto>>& outputs) {
  inputs.clear();
  for (auto& item : pir_parser.inputs) {
    auto value_info = MakeValueInfo(item);
    inputs.push_back(std::move(value_info));
  }
  outputs.clear();
  for (auto& item : pir_parser.outputs) {
    auto value_info = MakeValueInfo(item);
    outputs.push_back(std::move(value_info));
  }
}

void ModelExporter::ExportParameters(
    const PaddleParser& parser,
    std::vector<std::shared_ptr<ONNX_NAMESPACE::NodeProto>>& parameters) {
  parameters.clear();
  for (auto& item : parser.params) {
    // TODO(jiangjiajun) I'm not handling use_initializer now, but some day
    // I will
    auto node = MakeConstant(item.first, item.second);
    parameters.push_back(std::move(node));
  }
}

void ModelExporter::ExportParameters(
    const PaddlePirParser& pir_parser,
    std::vector<std::shared_ptr<ONNX_NAMESPACE::NodeProto>>& parameters) {
  parameters.clear();
  for (auto& item : pir_parser.params) {
    auto node = MakeConstant(item.first, item.second);
    parameters.push_back(std::move(node));
  }
}

ONNX_NAMESPACE::GraphProto ModelExporter::ExportIfBlock(
    PaddlePirParser& pir_parser, pir::Block& block) {
  std::vector<std::shared_ptr<ONNX_NAMESPACE::NodeProto>> temp_parameters;
  std::vector<std::shared_ptr<ONNX_NAMESPACE::ValueInfoProto>> temp_inputs;
  std::vector<std::shared_ptr<ONNX_NAMESPACE::ValueInfoProto>> temp_outputs;
  std::vector<pir::Operation*> sub_blocks_ops_copy(pir_parser.sub_blocks_ops);
  pir_parser.sub_blocks_ops.clear();
  for (auto& op : block.ops()) {
    if (op->name() != "builtin.parameter") {
      pir_parser.sub_blocks_ops.push_back(op);
    }
  }
  // generate sub-block op outputs names in GetMinOpSetVersion() function.
  // pir_parser.GetAllSubBlockOpOutputName(pir_parser.sub_blocks_ops);
  if (!pir_parser.sub_blocks_ops.empty()) {
    // get cf.yield op input
    pir::Operation* cf_yield_op = pir_parser.sub_blocks_ops.back();
    // std::vector<std::string> sub_block_outpus;
    for (int32_t idx = 0; idx < cf_yield_op->num_operands(); ++idx) {
      pir::Value value = cf_yield_op->operand(idx).source();
      auto cond_info = pir_parser.GetSubBlockValueTensorInfo(value);
      // sub_block_outpus.push_back(cond_info[0].name);
      temp_outputs.push_back(std::move(MakeValueInfo(cond_info[0])));
      if (value.defining_op() == nullptr) {
        value =
            pir::Value(pir_parser.while_op_values_args_map[&(*(value.impl()))]);
      }
      if (value.defining_op()->GetParent() != &block) {
        temp_inputs.push_back(std::move(MakeValueInfo(cond_info[0])));
      }
    }
  } else {
    // sub_blocks_ops is empty
    PADDLE_ENFORCE_NE(pir_parser.sub_blocks_ops.size(),
                      0,
                      ::common::errors::InvalidArgument(
                          "The number of ops of a control flow sub-block "
                          "cannot be zero."));
  }

  pir::Block* blockPtr = &block;
  auto graph = std::move(ExportBlock(pir_parser,
                                     blockPtr,
                                     temp_parameters,
                                     temp_inputs,
                                     temp_outputs,
                                     true,
                                     false));
  pir_parser.sub_blocks_ops.clear();
  pir_parser.sub_blocks_ops = sub_blocks_ops_copy;
  return graph;
}

ONNX_NAMESPACE::GraphProto ModelExporter::ExportBlock(
    PaddlePirParser& pir_parser,
    pir::Block* block,
    std::vector<std::shared_ptr<ONNX_NAMESPACE::NodeProto>>& parameters,
    std::vector<std::shared_ptr<ONNX_NAMESPACE::ValueInfoProto>>& inputs,
    std::vector<std::shared_ptr<ONNX_NAMESPACE::ValueInfoProto>>& outputs,
    bool if_in_subblock,
    bool is_while_block) {
  ONNX_NAMESPACE::GraphProto graph;
  graph.set_name("PaddlePaddle Graph in PIR mode");
  OnnxHelper temp_helper;
  std::vector<pir::Operation*> block_ops;
  for (auto& op : block->ops()) {
    if (op->name() != "builtin.parameter") {
      block_ops.push_back(op);
    }
  }
  auto num_ops = block_ops.size();
  temp_helper.nodes.reserve(num_ops * 3);
  temp_helper.Clear();
  for (auto i = 0; i < num_ops; ++i) {
    auto op = block_ops[i];
    if (op->name() == "pd_op.data" || op->name() == "pd_op.feed" ||
        op->name() == "pd_op.fetch" || op->name() == "cf.yield") {
      continue;
    }
    if (op->name() == "pd_op.if") {
      auto if_op = op->dyn_cast<paddle::dialect::IfOp>();
      // if branch graph
      pir::Block& true_block = if_op.true_block();
      auto then_graph = ExportIfBlock(pir_parser, true_block);
      // else branch graph
      pir::Block& false_block = if_op.false_block();
      auto else_graph = ExportIfBlock(pir_parser, false_block);
      // get if op input mask
      auto cond_info = pir_parser.GetTensorInfo(if_op.cond());
      auto cond_name = temp_helper.AutoCast(
          cond_info[0].name, cond_info[0].dtype, P2ODataType::BOOL);
      // get if op output
      auto num_results = if_op.num_results();
      std::vector<std::string> if_op_output_name;
      for (int i = 0; i < num_results; ++i) {
        auto value = if_op.result(i);
        auto out_info = pir_parser.GetTensorInfo(value);
        if_op_output_name.push_back(out_info[0].name);
      }
      auto node = temp_helper.MakeNode("If", {cond_name}, if_op_output_name);
      AddAttribute(node, "then_branch", then_graph);
      AddAttribute(node, "else_branch", else_graph);
      continue;
    }
    if (op->name() == "pd_op.while") {
      ExportWhile(pir_parser, &temp_helper, op);
      continue;
    }

    ExportOp(pir_parser,
             &temp_helper,
             opset_version_,
             op,
             i,
             if_in_subblock,
             verbose_);
  }
  if (if_in_subblock && !is_while_block) {
    for (auto& input_item : inputs) {
      for (int32_t idx = 0; idx < outputs.size(); ++idx) {
        auto output_item = outputs[idx];
        if (output_item->name() == input_item->name()) {
          output_item->set_name(pir_parser.GenOpInputOutputName("yield"));
          temp_helper.MakeNode(
              "Identity", {input_item->name()}, {output_item->name()});
          outputs[idx] = std::move(output_item);
        }
      }
    }
    inputs.clear();
  }
  for (auto& item : parameters) {
    *(graph.add_node()) = *(item.get());
  }

  for (auto& item : inputs) {
    *(graph.add_input()) = *(item.get());
  }

  for (auto& item : outputs) {
    *(graph.add_output()) = (*item.get());
  }

  for (auto& item : temp_helper.nodes) {
    *(graph.add_node()) = (*item.get());
  }

  for (auto& item : temp_helper.value_infos) {
    *(graph.add_value_info()) = (*item.get());
  }

  return std::move(graph);
}

ONNX_NAMESPACE::GraphProto ModelExporter::ExportBlock(
    const PaddleParser& parser,
    int32_t block_id,
    std::vector<std::shared_ptr<ONNX_NAMESPACE::NodeProto>>& parameters,
    std::vector<std::shared_ptr<ONNX_NAMESPACE::ValueInfoProto>>& inputs,
    std::vector<std::shared_ptr<ONNX_NAMESPACE::ValueInfoProto>>& outputs,
    OnnxHelper* helper,
    bool is_while_block) {
  ONNX_NAMESPACE::GraphProto graph;
  graph.set_name("PaddlePaddle Graph " + std::to_string(block_id));
  auto num_ops = parser.NumOfOps(block_id);

  // Init ONNXHelp
  OnnxHelper* temp_helper = nullptr;
  if (helper == nullptr) {
    temp_helper = new OnnxHelper();
    temp_helper->nodes.reserve(num_ops * 3);
    temp_helper->Clear();
  } else {
    temp_helper = helper;
  }

  for (auto op_id = 0; op_id < num_ops; ++op_id) {
    auto op = parser.GetOpDesc(block_id, op_id);
    if (op.type() == "feed") {
      continue;
    } else if (op.type() == "fetch") {
      continue;
    } else if (op.type() == "conditional_block") {
      auto out_info = parser.GetOpOutput(block_id, op_id, "Out");
      for (int index = 0; index < out_info.size(); index++) {
        sub_block_map_[out_info[index].name] = {block_id, op_id};
      }
      continue;
    } else if (op.type() == "select_input") {
      ExportSelectInput(parser, temp_helper, block_id, op_id);
      continue;
    } else if (op.type() == "fill_constant") {
      auto out_info = parser.GetOpOutput(block_id, op_id, "Out");
      sub_block_map_[out_info[0].name] = {block_id, op_id};
    } else if (op.type() == "while") {
      ExportWhile(parser, temp_helper, block_id, op_id);
      continue;
    }
    ExportOp(parser, temp_helper, opset_version_, block_id, op_id, verbose_);
  }

  ProcessGraphDumplicateNames(parameters,
                              inputs,
                              outputs,
                              temp_helper->nodes,
                              temp_helper->quantize_info,
                              is_while_block);

  // Process the model according to deploy_mackend_
  if (parser.is_quantized_model) {
    if (deploy_backend_ == "onnxruntime") {
      quantize_processer_ = new ORTQuantizeProcessor();
    } else if (deploy_backend_ == "rknn") {
      quantize_processer_ = new RKNNQuantizeProcessor();
    } else if (deploy_backend_ == "tensorrt") {
      quantize_processer_ = new TensorRTQuantizeProcessor();
    } else if (deploy_backend_ == "other") {
      quantize_processer_ = new OtherQuantizeProcessor();
    } else {
      Assert(false,
             "Only support onnxruntime/rknn/tensorrt/other as backend now, but "
             "now the backend is: " +
                 deploy_backend_ + ".");
    }
    P2OLogger() << "Deploy backend is: " << deploy_backend_ << std::endl;
    quantize_processer_->ProcessQuantizeModel(&parameters,
                                              &inputs,
                                              &outputs,
                                              &(temp_helper->nodes),
                                              temp_helper,
                                              parser,
                                              calibration_cache_);
    delete quantize_processer_;
    quantize_processer_ = nullptr;
    // Update int8 weights in quantized OP to float32
    UpdateParameters(temp_helper->updated_params, parameters);
  }

  for (auto& item : parameters) {
    *(graph.add_node()) = *(item.get());
  }

  for (auto& item : inputs) {
    *(graph.add_input()) = *(item.get());
  }

  for (auto& item : outputs) {
    *(graph.add_output()) = (*item.get());
  }

  for (auto& item : temp_helper->nodes) {
    *(graph.add_node()) = (*item.get());
  }

  for (auto& item : temp_helper->value_infos) {
    *(graph.add_value_info()) = (*item.get());
  }

  if (helper == nullptr) {
    delete temp_helper;
  }
  return std::move(graph);
}

void ModelExporter::UpdateParameters(
    const std::map<std::string, Weight>& params,
    std::vector<std::shared_ptr<ONNX_NAMESPACE::NodeProto>>& parameters) {
  for (auto& item : params) {
    auto node = MakeConstant(item.first, item.second);
    bool updated = false;
    for (int i = 0; i < parameters.size(); ++i) {
      auto old_node = parameters[i];
      if (old_node->output(0) == item.first) {
        parameters.erase(parameters.begin() + i);
        parameters.push_back(std::move(node));
        updated = true;
        break;
      }
    }
    if (!updated) {
      parameters.push_back(std::move(node));
    }
  }
}
void ModelExporter::ExportOp(const PaddlePirParser& pir_parser,
                             OnnxHelper* helper,
                             int32_t opset_version,
                             pir::Operation* op,
                             int64_t op_id,
                             bool if_in_subblock,
                             bool verbose) {
  auto mapper =
      MapperHelper::Get()->CreateMapper(convert_pir_op_name(op->name()),
                                        pir_parser,
                                        helper,
                                        op_id,
                                        if_in_subblock);
  mapper->deploy_backend = deploy_backend_;
  mapper->Run();
  delete mapper;
}

void ModelExporter::CovertCustomOps(const PaddleParser& parser,
                                    OnnxHelper* helper,
                                    int64_t block_id,
                                    int64_t op_id) {
  auto op = parser.GetOpDesc(block_id, op_id);
  std::vector<std::string> input_strs;
  for (auto i_index = 0; i_index < op.inputs_size(); i_index++) {
    auto input = op.inputs(i_index);
    std::string parameter = input.parameter();
    if (parser.OpHasInput(block_id, op_id, parameter)) {
      auto input_info = parser.GetOpInput(block_id, op_id, parameter);
      for (auto input : input_info) {
        input_strs.push_back(input.name);
        helper->MakeValueInfo(input.name, input.dtype, input.shape);
      }
    }
  }
  std::vector<std::string> output_strs;
  for (auto o_index = 0; o_index < op.outputs_size(); o_index++) {
    auto output = op.outputs(o_index);
    std::string parameter = output.parameter();
    if (parser.OpHasOutput(block_id, op_id, parameter)) {
      auto output_info = parser.GetOpOutput(block_id, op_id, parameter);
      for (auto output : output_info) {
        output_strs.push_back(output.name);
        helper->MakeValueInfo(output.name, output.dtype, output.shape);
      }
    }
  }
  auto node = helper->MakeNode(custom_ops[op.type()], input_strs, output_strs);
  node->set_domain("Paddle");
  for (auto attr_index = 0; attr_index < op.attrs_size(); attr_index++) {
    auto attr = op.attrs(attr_index);
    std::string attr_name = attr.name();
    if (attr_name == "op_callstack") {
      continue;
    }
    if (attr.has_i() || attr.has_l()) {
      int64_t val;
      parser.GetOpAttr(op, attr_name, &val);
      AddAttribute(node, attr_name, val);
    } else if (attr.has_f()) {
      float val;
      parser.GetOpAttr(op, attr_name, &val);
      AddAttribute(node, attr_name, val);
    } else if (attr.has_b()) {
      bool val;
      parser.GetOpAttr(op, attr_name, &val);
      AddAttribute(node, attr_name, static_cast<int64_t>(val));
    } else if (attr.has_s()) {
      std::string val;
      parser.GetOpAttr(op, attr_name, &val);
      AddAttribute(node, attr_name, val);
    } else if (attr.ints_size() > 0 || attr.longs_size() > 0) {
      std::vector<int64_t> vec;
      parser.GetOpAttr(op, attr_name, &vec);
      AddAttribute(node, attr_name, vec);
    } else if (attr.floats_size() > 0) {
      std::vector<float> vec;
      parser.GetOpAttr(op, attr_name, &vec);
      AddAttribute(node, attr_name, vec);
    } else if (attr.float64s_size() > 0) {
      std::vector<double> vec;
      parser.GetOpAttr(op, attr_name, &vec);
      std::vector<float> fp32_vec;
      for (auto val : vec) {
        fp32_vec.push_back(static_cast<float>(val));
      }
      AddAttribute(node, attr_name, fp32_vec);
    }
  }
  P2OLogger(true) << op.type() << " is exported as custom operator: "
                  << custom_ops[op.type()] << std::endl;
}

void ModelExporter::ExportOp(const PaddleParser& parser,
                             OnnxHelper* helper,
                             int32_t opset_version,
                             int64_t block_id,
                             int64_t op_id,
                             bool verbose) {
  auto op = parser.GetOpDesc(block_id, op_id);
  if (MapperHelper::Get()->IsRegistered(op.type())) {
    auto mapper = MapperHelper::Get()->CreateMapper(
        op.type(), parser, helper, block_id, op_id);
    mapper->deploy_backend = deploy_backend_;
    // Some operators will export as custom operator
    auto iter = custom_ops.find(op.type());
    if (iter != custom_ops.end()) {
      mapper->export_as_custom_op = true;
      mapper->custom_op_name = iter->second;
    }
    mapper->Run();
    delete mapper;
  } else if (custom_ops.find(op.type()) != custom_ops.end()) {
    CovertCustomOps(parser, helper, block_id, op_id);
  }
}

void ModelExporter::ProcessGraphDumplicateNames(
    std::vector<std::shared_ptr<ONNX_NAMESPACE::NodeProto>>& parameters,
    std::vector<std::shared_ptr<ONNX_NAMESPACE::ValueInfoProto>>& inputs,
    std::vector<std::shared_ptr<ONNX_NAMESPACE::ValueInfoProto>>& outputs,
    std::vector<std::shared_ptr<ONNX_NAMESPACE::NodeProto>>& nodes,
    std::map<std::string, QuantizeInfo>& quantize_info,
    bool is_while_block) {
  /********************* Create Tensor Names *********************/
  for (auto& item : nodes) {
    for (size_t i = 0; i < item->input_size(); ++i) {
      if (item->name().find("Loop") != std::string::npos) {
        // P2OLogger() << "nodes item input:" << item->input(i) << std::endl;
        while_tensor_names_.insert(item->input(i));
      }
    }
    for (size_t i = 0; i < item->output_size(); ++i) {
      if (item->name().find("Loop") != std::string::npos) {
        // P2OLogger() << "nodes item output:" << item->output(i) << std::endl;
        while_tensor_names_.insert(item->output(i));
      }
    }
  }
  // for (const auto& tensor_name : while_tensor_names_) {
  //   tensor_names_.erase(tensor_name);
  // }
  /********************* Create Tensor Names *********************/

  /********************* Rename *********************/
  for (auto& item : parameters) {
    for (size_t i = 0; i < item->output_size(); ++i) {
      if (tensor_names_.find(item->output(i)) != tensor_names_.end()) {
        P2OLogger()
            << "[WARNING] There's dumplicate names in exported parameters.";
        continue;
      }
      tensor_names_.insert(item->output(i));
    }
  }

  for (auto& item : inputs) {
    tensor_names_.insert(item->name());
  }
  std::map<std::string, std::string> renamer;
  for (auto& item : nodes) {
    // update node inputs
    for (size_t i = 0; i < item->input_size(); ++i) {
      if (renamer.find(item->input(i)) != renamer.end()) {
        auto updated_name = renamer[item->input(i)];
        while (renamer.find(updated_name) != renamer.end()) {
          updated_name = renamer[updated_name];
        }
        *(item->mutable_input(i)) = updated_name;
      }
    }

    // if there's dumplicate name , it will generate new name and replace
    // the dumplicate name
    for (size_t i = 0; i < item->output_size(); ++i) {
      if (tensor_names_.find(item->output(i)) != tensor_names_.end()) {
        if (is_while_block) {
          if (while_tensor_names_.find(item->output(i)) !=
              while_tensor_names_.end()) {
            // P2OLogger() << "Skip: " << item->output(i) << std::endl;
            continue;
          }
        }
        std::string renamed_tensor_name = item->output(i);
        while (renamer.find(renamed_tensor_name) != renamer.end()) {
          renamed_tensor_name = renamer[renamed_tensor_name];
        }
        auto new_tensor_name =
            MapperHelper::Get()->GenName(renamed_tensor_name);
        if (quantize_info.find(renamed_tensor_name) != quantize_info.end()) {
          quantize_info[new_tensor_name] = quantize_info[renamed_tensor_name];
        }
        *(item->mutable_output(i)) = new_tensor_name;
        renamer[renamed_tensor_name] = new_tensor_name;
      }
      tensor_names_.insert(item->output(i));
    }
  }

  for (auto& item : outputs) {
    if (renamer.find(item->name()) != renamer.end()) {
      auto updated_name = renamer[item->name()];
      while (renamer.find(updated_name) != renamer.end()) {
        updated_name = renamer[updated_name];
      }
      item->set_name(updated_name);
    }
  }
  /********************* Rename *********************/
}

void ModelExporter::SaveExternalData(::ONNX_NAMESPACE::GraphProto* graph,
                                     const std::string& external_file_path,
                                     bool* save_external) {
  P2OLogger() << "The exported ONNX model is bigger than 2G, external data "
                 "will save to file: "
              << external_file_path << std::endl;
  std::string file_name = GetFilenameFromPath(external_file_path);
  if (save_external) {
    *save_external = true;
  }
  std::fstream f(external_file_path, std::ios::out);
  Assert(
      f.is_open(),
      "Failed to open: " + external_file_path + " file to save external data");
  for (auto index = 0; index < graph->node_size(); index++) {
    auto node = graph->mutable_node(index);
    if (node->op_type() != "Constant") {
      continue;
    }
    for (auto i = 0; i < node->attribute_size(); i++) {
      auto attr = node->mutable_attribute(i);
      if (attr->name() != "value") {
        continue;
      }
      auto tensor = attr->mutable_t();

      if (tensor->raw_data().size() <= 128) {
        continue;
      }

      tensor->set_data_location(ONNX_NAMESPACE::TensorProto::EXTERNAL);
      auto external_data = tensor->add_external_data();
      external_data->set_key("location");
      external_data->set_value(file_name);

      external_data = tensor->add_external_data();
      external_data->set_key("offset");
      f.seekg(0, std::ios::end);
      int64_t offset = f.tellg();
      external_data->set_value(std::to_string(offset));
      auto raw_data = tensor->raw_data();
      f << raw_data;
      external_data = tensor->add_external_data();
      external_data->set_key("length");
      int64_t raw_datas_size = raw_data.size();
      external_data->set_value(std::to_string(raw_datas_size));
      tensor->clear_raw_data();
    }
  }
  f.close();
}
void ModelExporter::ONNXChecker(const ONNX_NAMESPACE::ModelProto& model,
                                const bool& verbose) {
  // TODO(jiangjiajun)
  // If we need to integrate with framework
  // this check will return a information
  // to let framework know the conversion is
  // pass or fail
  try {
    // ONNX_NAMESPACE::checker::check_model(*(model.get()));
    ONNX_NAMESPACE::checker::check_model(model);
  } catch (const std::exception& e) {
    P2OLogger(verbose) << "The exported ONNX model is invalid." << std::endl;
    P2OLogger(verbose) << "Model checker error log: " << e.what() << std::endl;
  }
  P2OLogger(verbose) << "PaddlePaddle model is exported as ONNX format now."
                     << std::endl;
}

std::string ModelExporter::Run(PaddlePirParser& pir_parser,
                               int opset_version,
                               bool auto_upgrade_opset,
                               bool verbose,
                               bool enable_onnx_checker,
                               bool enable_experimental_op,
                               bool enable_optimize,
                               const std::string& deploy_backend,
                               std::string* calibration_cache,
                               const std::string& external_file,
                               bool* save_external,
                               bool export_fp16_model,
                               std::vector<std::string> disable_fp16_op_types) {
  verbose_ = verbose;
  deploy_backend_ = deploy_backend;
  calibration_cache_ = calibration_cache;
  // Clear name_counter, this use to generate unique name for intermdiate
  // while converting all the op
  MapperHelper::Get()->ClearNameCounter();
  if (!IsOpsRegistered(pir_parser, enable_experimental_op)) {
    Assert(false,
           "Due to the unsupported operators, the conversion is aborted.");
  }
  // Set ONNX Opset Version
  opset_version_ = opset_version;
  SetOpsetVersion(pir_parser, auto_upgrade_opset);
  // Set ONNX IR Version
  SetIRVersion();
  // Export Parser Parameters
  std::vector<std::shared_ptr<ONNX_NAMESPACE::NodeProto>> parameters;
  ExportParameters(pir_parser, parameters);
  // Export Parser Inputs and Outputs
  std::vector<std::shared_ptr<ONNX_NAMESPACE::ValueInfoProto>> inputs;
  std::vector<std::shared_ptr<ONNX_NAMESPACE::ValueInfoProto>> outputs;
  ExportInputOutputs(pir_parser, inputs, outputs);
  // Export Blocks
  auto share_graph = ExportBlock(pir_parser,
                                 pir_parser.pir_program_->block(),
                                 parameters,
                                 inputs,
                                 outputs,
                                 false,
                                 false);
  *onnx_model_.mutable_graph() = share_graph;
  if (enable_onnx_checker) {
    ONNXChecker(onnx_model_, verbose);
  }
  std::string out;
  if (!onnx_model_.SerializeToString(&out)) {
    P2OLogger(verbose)
        << "Error happenedd while optimizing the exported ONNX model."
        << std::endl;
    return "";
  }
  return out;
}

std::string ModelExporter::Run(const PaddleParser& parser,
                               int opset_version,
                               bool auto_upgrade_opset,
                               bool verbose,
                               bool enable_onnx_checker,
                               bool enable_experimental_op,
                               bool enable_optimize,
                               const std::string& deploy_backend,
                               std::string* calibration_cache,
                               const std::string& external_file,
                               bool* save_external,
                               bool export_fp16_model,
                               std::vector<std::string> disable_fp16_op_types) {
  verbose_ = verbose;
  deploy_backend_ = deploy_backend;
  calibration_cache_ = calibration_cache;

  // Clear name_counter, this use to generate unique name for intermdiate
  // while converting all the op
  MapperHelper::Get()->ClearNameCounter();

  if (!IsOpsRegistered(parser, enable_experimental_op)) {
    Assert(false,
           "Due to the unsupported operators, the conversion is aborted.");
  }

  // Set ONNX Opset Version
  opset_version_ = opset_version;
  SetOpsetVersion(parser, auto_upgrade_opset);

  // Set ONNX IR Version
  SetIRVersion();

  // Export Parser Parameters
  std::vector<std::shared_ptr<ONNX_NAMESPACE::NodeProto>> parameters;
  ExportParameters(parser, parameters);
  // Export Parser Inputs and Outputs
  std::vector<std::shared_ptr<ONNX_NAMESPACE::ValueInfoProto>> inputs;
  std::vector<std::shared_ptr<ONNX_NAMESPACE::ValueInfoProto>> outputs;
  ExportInputOutputs(parser, inputs, outputs);

  auto share_graph = ExportBlock(parser, 0, parameters, inputs, outputs);
  *onnx_model_.mutable_graph() = share_graph;

  if (enable_optimize) {
    onnx_model_ = Optimize(onnx_model_);
  }

  // convert fp32 model to fp16
  if (export_fp16_model) {
    P2OLogger(verbose) << "Convert FP32 ONNX model to FP16." << std::endl;
    ConvertFp32ToFp16 convert;
    convert.SetCustomOps(custom_ops);
    convert.AddDisabledOpTypes(disable_fp16_op_types);
    convert.Convert(&onnx_model_);
  }

  // save external data file for big model
  std::string external_data_file;
  if (onnx_model_.ByteSizeLong() > INT_MAX) {
    if (external_file.empty()) {
      external_data_file = "external_data";
    } else {
      external_data_file = external_file;
    }
  }

  if (external_data_file.size()) {
    SaveExternalData(
        onnx_model_.mutable_graph(), external_data_file, save_external);
  }

  // check model
  if (enable_onnx_checker) {
    ONNXChecker(onnx_model_, verbose);
  }

  std::string out;
  if (!onnx_model_.SerializeToString(&out)) {
    P2OLogger(verbose)
        << "Error happenedd while optimizing the exported ONNX model."
        << std::endl;
    return "";
  }
  return out;
}

ONNX_NAMESPACE::ModelProto ModelExporter::Optimize(
    const ONNX_NAMESPACE::ModelProto& model) {
  ONNX_NAMESPACE::optimization::Optimizer::passes
      .registerPass<ONNX_NAMESPACE::optimization::FuseConstantReshape>();
  ONNX_NAMESPACE::optimization::Optimizer::passes
      .registerPass<ONNX_NAMESPACE::optimization::FuseConstantUnsqueeze>();
  ONNX_NAMESPACE::optimization::Optimizer::passes
      .registerPass<ONNX_NAMESPACE::optimization::FusePaddleConvBias>();
  ONNX_NAMESPACE::optimization::Optimizer::passes
      .registerPass<ONNX_NAMESPACE::optimization::FuseUnsqueezeConv2dSqueeze>();
  ONNX_NAMESPACE::optimization::Optimizer::passes
      .registerPass<ONNX_NAMESPACE::optimization::EliminateNonTranspose>();
  ONNX_NAMESPACE::optimization::Optimizer::passes
      .registerPass<ONNX_NAMESPACE::optimization::FuseConstantCast>();
  std::vector<std::string> passes = {"eliminate_identity",
                                     "eliminate_deadend",
                                     "eliminate_deadend",
                                     "fuse_constant_reshape",
                                     "fuse_constant_unsqueeze",
                                     "fuse_paddle_conv_bias",
                                     "fuse_consecutive_transposes",
                                     "eliminate_non_transpose",
                                     "fuse_matmul_add_bias_into_gemm",
                                     "eliminate_identity",
                                     "eliminate_deadend",
                                     "eliminate_unused_initializer"};
  return ONNX_NAMESPACE::optimization::Optimize(model, passes);
}
}  // namespace paddle2onnx
