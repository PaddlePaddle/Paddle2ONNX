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
#include "paddle/fluid/pir/dialect/operator/ir/control_flow_op.h"
namespace paddle2onnx {
void ModelExporter::ExportWhile(PaddlePirParser& pir_parser,OnnxHelper* temp_helper,pir::Operation* op) {
  // ================================
  //  construct loop body sub graph
  // ================================
  std::vector<TensorInfo> inputs_info;
  std::vector<TensorInfo> outputs_info;
  auto while_op = op->dyn_cast<paddle::dialect::WhileOp>();
  auto cond_info = pir_parser.GetTensorInfo(while_op.cond());
  // mapping args and inputs in while op using while_op_input_value_map
  std::vector<pir::detail::ValueImpl*> while_op_input_value_address;
  std::vector<pir::detail::ValueImpl*> while_op_input_arg_address;
  pir_parser.while_op_input_value_map.clear(); // wangmingkai02: handle nested loop situations in future.

  // record input value address
  for(int index = 1; index < while_op.num_operands(); index++){
    const pir::Value& value = while_op.operand_source(index);
    inputs_info.push_back(pir_parser.GetTensorInfo(pir_parser.GetOpOutputName(value), value.type()));
    while_op_input_value_address.push_back(&(*(value).impl())); // get value address
  }
  // record args value address
  std::vector<pir::Value> args = while_op.block_args();
  for(int i = 0; i< args.size(); i++){
    const pir::Value& value = args[i];
    while_op_input_arg_address.push_back(&(*(value.impl())));
  }

  // mapping
  for(int index=0; index < while_op_input_value_address.size(); index++){
    pir_parser.while_op_input_value_map[while_op_input_arg_address[index]] = while_op_input_value_address[index];
  }

  std::vector<pir::Operation*> sub_blocks_ops_copy(pir_parser.sub_blocks_ops);
  pir_parser.sub_blocks_ops.clear();
  auto& body_block = while_op.body();
  for (auto& op : body_block.ops()) {
    if (op->name() != "builtin.parameter") {
      pir_parser.sub_blocks_ops.push_back(op);
    }
  }

  pir_parser.GetAllSubBlockOpOutputName(pir_parser.sub_blocks_ops);
  if (!pir_parser.sub_blocks_ops.empty()) {
    // get cf.yeild op input
    pir::Operation* cf_yield_op = pir_parser.sub_blocks_ops.back();
    PADDLE_ENFORCE_EQ(cf_yield_op->name(),
                      "cf.yield",
                      ::common::errors::InvalidArgument(
                        "The last op of a control flow sub-block must be cf.yield"));
    for (auto oprand : cf_yield_op->operands()) {
      pir::Value value = oprand.source();
      auto info = pir_parser.GetSubBlockValueTensorInfo(value);
      outputs_info.push_back(info[0]);
    }

  } else {
    // sub_blocks_ops is empty
    PADDLE_ENFORCE_NE(pir_parser.sub_blocks_ops.size(),
                      0,
                      ::common::errors::InvalidArgument(
                          "The number of ops of a control flow sub-block "
                          "cannot be zero."));
  }

  ONNX_NAMESPACE::GraphProto graph;
  std::vector<std::shared_ptr<ONNX_NAMESPACE::NodeProto>> parameters;
  std::vector<std::shared_ptr<ONNX_NAMESPACE::ValueInfoProto>> inputs;
  std::vector<std::shared_ptr<ONNX_NAMESPACE::ValueInfoProto>> outputs;
  auto iter_name = MapperHelper::Get()->GenName("loop.iter");
  TensorInfo iter_info(iter_name, std::vector<int64_t>(1, 1),
                      P2ODataType::INT64);
  // inputs
  inputs.push_back(std::move(MakeValueInfo(iter_info)));
  inputs.push_back(std::move(MakeValueInfo(cond_info[0])));
  for (size_t i = 0; i < inputs_info.size(); ++i) {
    inputs.push_back(std::move(MakeValueInfo(inputs_info[i])));
  }
  // outputs
  for (size_t i = 0; i < outputs_info.size(); ++i) {
    outputs.push_back(std::move(MakeValueInfo(outputs_info[i])));
  }
  pir::Block* blockPtr = &body_block;
  graph = ExportBlock(pir_parser, blockPtr, parameters, inputs, outputs, true, true);
  pir_parser.sub_blocks_ops.clear();
  pir_parser.sub_blocks_ops = sub_blocks_ops_copy;

  // =====================
  //  construct loop node
  // =====================
  std::vector<std::string> input_names;
  std::vector<std::string> output_names;
  input_names.push_back(""); // skip max loop iter
  input_names.push_back(cond_info[0].name);
  for(size_t i = 0; i < inputs_info.size(); ++i) {
    input_names.push_back(inputs_info[i].name);
  }
  for(size_t i = 0; i < op->num_results(); i++) {
    output_names.push_back(pir_parser.GetOpOutputName(op->result(i)));
  }
  auto loop_node = temp_helper->MakeNode("Loop", input_names, output_names);
  AddAttribute(loop_node, "body", graph);
}
}  // namespace paddle2onnx
