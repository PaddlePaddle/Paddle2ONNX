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

#include <google/protobuf/message.h>
#include <onnx/checker.h>

#include <array>

#include "onnxoptimizer/optimize.h"
#include "paddle/fluid/pir/dialect/operator/ir/control_flow_op.h"
#include "paddle/phi/core/enforce.h"
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
  for (auto op : pir_parser.global_blocks_ops) {
    if (op->name() == "pd_op.data" || op->name() == "pd_op.fetch") {
      continue;
    }
    if (op->name() == "pd_op.if") {
      continue;
    }
    if (op->name() == "pd_op.while") {
      continue;
    }
    std::string op_name = convert_pir_op_name(op->name());
    if (!MapperHelper::Get()->IsRegistered(op_name)) {
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

bool ModelExporter::IsOpsRegistered(const PaddleParser& parser,
                                    bool enable_experimental_op) {
  OnnxHelper temp_helper;
  std::set<std::string> unsupported_ops;
  for (auto i = 0; i < parser.NumOfBlocks(); ++i) {
    for (auto j = 0; j < parser.NumOfOps(i); ++j) {
      auto op = parser.GetOpDesc(i, j);
      if (op.type() == "feed" || op.type() == "fetch") {
        continue;
      }

      if (op.type() == "conditional_block" || op.type() == "select_input") {
        continue;
      }
#if 0
        if (op.type() == "while" && enable_experimental_op) {
          if (!IsLoopSupported(parser, i, j)) {
            unsupported_ops.insert("while");
          }
          continue;
        }
#endif
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

  auto logger = P2OLogger();
  logger << "Oops, there are some operators not supported yet, including ";
  for (auto& item : unsupported_ops) {
    logger << item << ",";
  }
  logger << std::endl;
  return (unsupported_ops.size() == 0);
}

int32_t ModelExporter::GetMinOpsetVersion(const PaddleParser& parser) {
  int32_t max_opset = 7;
  std::set<std::string> verbose_log;
  OnnxHelper helper;
  for (auto i = 0; i < parser.NumOfBlocks(); ++i) {
    for (auto j = 0; j < parser.NumOfOps(i); ++j) {
      auto op = parser.GetOpDesc(i, j);

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
      } else {
        auto mapper =
            MapperHelper::Get()->CreateMapper(op.type(), parser, &helper, i, j);
        current_opset = mapper->GetMinOpsetVersion(verbose_);
        delete mapper;
      }
#if 0
        if (op.type() == "while") {
          P2OLogger() << "Detected control flow 'while' op in your model, "
                      << "this requires the minimal opset version of 13."
                      << std::endl;
          current_opset = 13;
        }
#endif

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

int32_t ModelExporter::GetMinOpsetVersion(const PaddlePirParser& pir_parser) {
  int32_t max_opset = 7;
  std::set<std::string> verbose_log;
  OnnxHelper helper;
  for (auto i = 0; i < pir_parser.global_blocks_ops.size(); i++) {
    std::string op_name = pir_parser.global_blocks_ops[i]->name();
    if (op_name == "pd_op.data" || op_name == "pd_op.fetch") {
      continue;
    }
    if (op_name == "pd_op.if" || op_name == "pd_op.while") {
      continue;
    }
    int current_opset = 7;
    auto mapper = MapperHelper::Get()->CreateMapper(
        convert_pir_op_name(op_name), pir_parser, &helper, i, false);
    current_opset = mapper->GetMinOpsetVersion(verbose_);
    delete mapper;

    // TODO : some bugs will appear, not solved yet
    // if (current_opset > max_opset) {
    //   max_opset = current_opset;
    //   if (current_opset > opset_version_) {
    //     verbose_log.insert("Due to the operator: " +
    //                         pir_parser.global_blocks_ops[i]->name() + ",
    //                         " + "requires opset_version >= " +
    //                         std::to_string(current_opset) + ".");
    //   }
    // }
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
  int32_t min_opset = GetMinOpsetVersion(pir_parser);
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
  pir_parser.GetAllSubBlockOpOutputName(pir_parser.sub_blocks_ops);
  if (!pir_parser.sub_blocks_ops.empty()) {
    // get cf.yeild op input
    pir::Operation* cf_yield_op = pir_parser.sub_blocks_ops.back();
    // std::vector<std::string> sub_block_outpus;
    for (auto oprand : cf_yield_op->operands()) {
      pir::Value value = oprand.source();
      auto cond_info = pir_parser.GetSubBlockValueTensorInfo(value);
      // sub_block_outpus.push_back(cond_info[0].name);
      temp_outputs.push_back(std::move(MakeValueInfo(cond_info[0])));
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
  auto graph = std::move(ExportBlock(
      pir_parser, blockPtr, temp_parameters, temp_inputs, temp_outputs, true, false));
  pir_parser.sub_blocks_ops.clear();
  pir_parser.sub_blocks_ops = sub_blocks_ops_copy;
  return graph;
}

ONNX_NAMESPACE::GraphProto ModelExporter::ExportConditionalBlock(
    const PaddleParser& parser,
    int32_t block_id,
    int32_t op_id,
    const std::string& output_names) {
  auto op = parser.GetOpDesc(block_id, op_id);

  // Get sub_block_idx
  int32_t sub_block_idx = -1;
  for (size_t i = 0; i < op.attrs_size(); ++i) {
    if (op.attrs(i).name() == "sub_block") {
      sub_block_idx = op.attrs(i).block_idx();
      break;
    }
  }
  Assert(sub_block_idx != -1,
         "Due to the unsupported sub_block_idx, the conversion is aborted.");

  std::vector<std::shared_ptr<ONNX_NAMESPACE::NodeProto>> temp_parameters;

  std::vector<std::shared_ptr<ONNX_NAMESPACE::ValueInfoProto>> temp_inputs;

  std::vector<std::shared_ptr<ONNX_NAMESPACE::ValueInfoProto>> temp_outputs;
  auto out_info = parser.GetOpOutput(block_id, op_id, "Out");
  for (int index = 0; index < out_info.size(); index++) {
    if (out_info[index].name != output_names) {
      continue;
    }
    temp_outputs.push_back(std::move(MakeValueInfo(out_info[index])));
  }
  return std::move(ExportBlock(
      parser, sub_block_idx, temp_parameters, temp_inputs, temp_outputs));
}

ONNX_NAMESPACE::GraphProto ModelExporter::ExportBlock(
    PaddlePirParser& pir_parser,
    pir::Block* block,
    std::vector<std::shared_ptr<ONNX_NAMESPACE::NodeProto>>& parameters,
    std::vector<std::shared_ptr<ONNX_NAMESPACE::ValueInfoProto>>& inputs,
    std::vector<std::shared_ptr<ONNX_NAMESPACE::ValueInfoProto>>& outputs,
    bool if_in_subblock, bool is_while_block) {
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
    if (op->name() == "pd_op.data" || op->name() == "pd_op.fetch" || op->name() == "cf.yield") {
      continue;
    }
    if (op->name() == "pd_op.full_int_array") { // this is a trick
      bool needExport = false;
      for (auto it = op->result(0).use_begin(); it != op->result(0).use_end();
           ++it) {
        if (!(it->owner()->name() == "pd_op.pool2d")) {
          needExport = true;
          break;
        }
      }
      if (!needExport) continue;
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
      ExportWhile(pir_parser,&temp_helper,op);
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
    std::vector<std::shared_ptr<ONNX_NAMESPACE::ValueInfoProto>>& outputs) {
  ONNX_NAMESPACE::GraphProto graph;
  graph.set_name("PaddlePaddle Graph " + std::to_string(block_id));
  OnnxHelper temp_helper;
  auto num_ops = parser.NumOfOps(block_id);
  temp_helper.nodes.reserve(num_ops * 3);
  temp_helper.Clear();
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
      // 如果找到，则输出对应的值；否则输出错误信息
      // 遍历输入Tensor
      auto input_info = parser.GetOpInput(block_id, op_id, "X");

      Assert(input_info.size() == 2,
             "Only support when number of select_input's input_node is 2.");

      // 构建 else 分支图
      auto else_node_name = input_info[0].name;
      auto conditional_block_cood_it = sub_block_map_.find(else_node_name);
      Assert(conditional_block_cood_it != sub_block_map_.end(),
             "Don't find select_input else_input node.");
      auto conditional_block_cood = conditional_block_cood_it->second;
      auto else_graph = ExportConditionalBlock(parser,
                                               conditional_block_cood.first,
                                               conditional_block_cood.second,
                                               else_node_name);

      // 构建 then 分支图
      auto then_node_name = input_info[1].name;
      conditional_block_cood_it = sub_block_map_.find(then_node_name);
      Assert(conditional_block_cood_it != sub_block_map_.end(),
             "Don't find select_input then_input node.");
      conditional_block_cood = conditional_block_cood_it->second;
      auto then_graph = ExportConditionalBlock(parser,
                                               conditional_block_cood.first,
                                               conditional_block_cood.second,
                                               then_node_name);

      auto cond_info = parser.GetOpInput(block_id, op_id, "Mask");
      auto output_info = parser.GetOpOutput(block_id, op_id, "Out");
      auto cond_name = temp_helper.AutoCast(
          cond_info[0].name, cond_info[0].dtype, P2ODataType::BOOL);
      auto node =
          temp_helper.MakeNode("If", {cond_name}, {output_info[0].name});
      AddAttribute(node, "then_branch", then_graph);
      AddAttribute(node, "else_branch", else_graph);
      continue;
    }
    ExportOp(parser, &temp_helper, opset_version_, block_id, op_id, verbose_);
  }

  ProcessGraphDumplicateNames(parameters,
                              inputs,
                              outputs,
                              temp_helper.nodes,
                              temp_helper.quantize_info);
  if (parser.is_quantized_model) {
    quantize_model_processer.ProcessQuantizeModel(&parameters,
                                                  &inputs,
                                                  &outputs,
                                                  &temp_helper.nodes,
                                                  &temp_helper,
                                                  deploy_backend_,
                                                  parser,
                                                  calibration_cache_);
    // Update int8 weights in quantized OP to float32
    UpdateParameters(temp_helper.updated_params, parameters);
  }

  for (auto &item : parameters) {
    *(graph.add_node()) = *(item.get());
  }

  for (auto &item : inputs) {
    *(graph.add_input()) = *(item.get());
  }

  for (auto &item : outputs) {
    *(graph.add_output()) = (*item.get());
  }

  for (auto &item : temp_helper.nodes) {
    *(graph.add_node()) = (*item.get());
  }

  for (auto &item : temp_helper.value_infos) {
    *(graph.add_value_info()) = (*item.get());
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

void ModelExporter::ExportOp(const PaddleParser& parser,
                             OnnxHelper* helper,
                             int32_t opset_version,
                             int64_t block_id,
                             int64_t op_id,
                             bool verbose) {
  auto op = parser.GetOpDesc(block_id, op_id);
#if 0
    if (op.type() == "while")
    {
      return ExportLoop(parser, helper, opset_version, block_id, op_id, verbose);
    }
#endif
  auto mapper = MapperHelper::Get()->CreateMapper(
      op.type(), parser, helper, block_id, op_id);
  mapper->deploy_backend = deploy_backend_;
  mapper->Run();
  delete mapper;
}

void ModelExporter::ProcessGraphDumplicateNames(
    std::vector<std::shared_ptr<ONNX_NAMESPACE::NodeProto>> &parameters,
    std::vector<std::shared_ptr<ONNX_NAMESPACE::ValueInfoProto>> &inputs,
    std::vector<std::shared_ptr<ONNX_NAMESPACE::ValueInfoProto>> &outputs,
    std::vector<std::shared_ptr<ONNX_NAMESPACE::NodeProto>> &nodes,
    std::map<std::string, QuantizeInfo> &quantize_info) {
  std::map<std::string, std::string> renamer;
  for (auto& item : parameters) {
    for (size_t i = 0; i < item->output_size(); ++i) {
      if (tensor_names_.find(item->output(i)) != tensor_names_.end()) {
        Assert(false, "There's dumplicate names in exported parameters.");
      }
      tensor_names_.insert(item->output(i));
    }
  }

  for (auto& item : inputs) {
    if (tensor_names_.find(item->name()) != tensor_names_.end()) {
      continue;
      // Assert(false, "There's dumplicate names:" + item->name() + " in
      // exported parameters and inputs.");
    }
    tensor_names_.insert(item->name());
  }

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
        std::string renamed_tensor_name = item->output(i);
        while (renamer.find(renamed_tensor_name) != renamer.end()) {
          renamed_tensor_name = renamer[renamed_tensor_name];
        }
        auto new_tensor_name =
            MapperHelper::Get()->GenName(renamed_tensor_name);
        P2OLogger() << "Find dumplicate output name '" << renamed_tensor_name
                    << "', it will rename to '" << new_tensor_name << "'."
                    << std::endl;
        if (quantize_info.find(renamed_tensor_name) != quantize_info.end()) {
          quantize_info[new_tensor_name] = quantize_info[renamed_tensor_name];
        }
        *(item->mutable_output(i)) = new_tensor_name;
        renamer[renamed_tensor_name] = new_tensor_name;
      }
      tensor_names_.insert(item->output(i));
    }
  }

  for (auto &item : outputs) {
    if (renamer.find(item->name()) != renamer.end()) {
      auto updated_name = renamer[item->name()];
      while (renamer.find(updated_name) != renamer.end()) {
        updated_name = renamer[updated_name];
      }
      item->set_name(updated_name);
    }
  }
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
  tensor_names_.clear();
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
  // Export Blocks
  tensor_names_.clear();

  auto share_graph = ExportBlock(parser, 0, parameters, inputs, outputs);
  *onnx_model_.mutable_graph() = share_graph;

  if (enable_optimize) {
    onnx_model_ = Optimize(onnx_model_);
  }

  // convert fp32 model to fp16
  if (export_fp16_model) {
    P2OLogger(verbose) << "Convert FP32 ONNX model to FP16." << std::endl;
    ConvertFp32ToFp16 convert;
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
