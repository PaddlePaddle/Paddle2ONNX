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

#include "paddle/fluid/pir/dialect/operator/ir/control_flow_op.h"
#include "paddle2onnx/mapper/exporter.h"
namespace paddle2onnx {
void ModelExporter::ExportWhile(PaddlePirParser& pir_parser,
                                OnnxHelper* temp_helper,
                                pir::Operation* op) {
  // ================================
  //  construct loop body sub graph
  // ================================
  std::vector<TensorInfo> inputs_info;
  std::vector<TensorInfo> outputs_info;
  std::vector<std::shared_ptr<ONNX_NAMESPACE::NodeProto>> extra_nodes;
  auto while_op = op->dyn_cast<paddle::dialect::WhileOp>();
  auto cond_info = pir_parser.GetTensorInfo(while_op.cond());
  std::unordered_set<std::string> names;
  for (int index = 1; index < while_op.num_operands(); index++) {
    const pir::Value& value = while_op.operand_source(index);
    std::string name = pir_parser.GetSubBlockOpOutputName(value);
    if (names.count(name)) {
      // there are duplicated varianble names in while op's operands.
      name = temp_helper->MakeNode("Identity", {name})->output(0);
      pir_parser.while_op_args_name_map[&(
          *((while_op.block_args()[index - 1]).impl()))] = name;
    } else {
      names.insert(name);
    }
    inputs_info.push_back(pir_parser.GetTensorInfo(name, value.type()));
  }
  pir_parser.GetWhileInputValuesAndArgsMappings(&while_op);

  std::vector<pir::Operation*> sub_blocks_ops_copy(pir_parser.sub_blocks_ops);
  pir_parser.sub_blocks_ops.clear();
  auto& body_block = while_op.body();
  for (auto& op : body_block.ops()) {
    if (op->name() != "builtin.parameter") {
      pir_parser.sub_blocks_ops.push_back(op);
    }
  }

  // generate sub-block op outputs names in GetMinOpSetVersion() function.
  // pir_parser.GetAllSubBlockOpOutputName(pir_parser.sub_blocks_ops);
  if (!pir_parser.sub_blocks_ops.empty()) {
    // get cf.yeild op input
    pir::Operation* cf_yield_op = pir_parser.sub_blocks_ops.back();
    PADDLE_ENFORCE_EQ(
        cf_yield_op->name(),
        "cf.yield",
        ::common::errors::InvalidArgument(
            "The last op of a control flow sub-block must be cf.yield"));
    for (auto oprand : cf_yield_op->operands()) {
      pir::Value value = oprand.source();
      if (value.defining_op()->GetParent() != cf_yield_op->GetParent()) {
        std::string name = pir_parser.GetSubBlockOpOutputName(value);
        auto node = std::make_shared<ONNX_NAMESPACE::NodeProto>();
        auto node_name = MapperHelper::Get()->GenName("Identity");
        node->set_name(node_name);
        node->set_op_type("Identity");
        node->add_input(name);
        node->add_output(MapperHelper::Get()->GenName("Identity"));
        extra_nodes.push_back(node);
        TensorInfo info =
            pir_parser.GetTensorInfo(node->output(0), value.type());
        outputs_info.push_back(info);
      } else {
        auto info = pir_parser.GetSubBlockValueTensorInfo(value);
        outputs_info.push_back(info[0]);
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

  ONNX_NAMESPACE::GraphProto graph;
  std::vector<std::shared_ptr<ONNX_NAMESPACE::NodeProto>> parameters;
  std::vector<std::shared_ptr<ONNX_NAMESPACE::ValueInfoProto>> inputs;
  std::vector<std::shared_ptr<ONNX_NAMESPACE::ValueInfoProto>> outputs;
  auto iter_name = MapperHelper::Get()->GenName("loop.iter");
  TensorInfo iter_info(
      iter_name, std::vector<int64_t>(1, 1), P2ODataType::INT64);
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
  graph = ExportBlock(
      pir_parser, blockPtr, parameters, inputs, outputs, true, true);
  for (auto& item : extra_nodes) {
    *(graph.add_node()) = (*item.get());
  }

  pir_parser.sub_blocks_ops.clear();
  pir_parser.sub_blocks_ops = sub_blocks_ops_copy;

  // =====================
  //  construct loop node
  // =====================
  std::vector<std::string> input_names;
  std::vector<std::string> output_names;
  input_names.push_back("");  // skip max loop iter
  input_names.push_back(cond_info[0].name);
  for (size_t i = 0; i < inputs_info.size(); ++i) {
    input_names.push_back(inputs_info[i].name);
  }
  for (size_t i = 0; i < op->num_results(); i++) {
    output_names.push_back(pir_parser.GetSubBlockOpOutputName(op->result(i)));
  }
  auto loop_node = temp_helper->MakeNode("Loop", input_names, output_names);
  AddAttribute(loop_node, "body", graph);
}

void ModelExporter::ExportWhile(const PaddleParser& parser,
                                OnnxHelper* temp_helper,
                                int32_t block_id,
                                int32_t op_id) {
  auto op = parser.GetOpDesc(block_id, op_id);
  auto x_info = parser.GetOpInput(block_id, op_id, "X");
  auto cond_info = parser.GetOpInput(block_id, op_id, "Condition");
  auto out_info = parser.GetOpOutput(block_id, op_id, "Out");

  ONNX_NAMESPACE::GraphProto graph;
  /********************* Creat Body Gragh *********************/
  int32_t sub_block_idx = -1;
  for (size_t i = 0; i < op.attrs_size(); ++i) {
    if (op.attrs(i).name() == "sub_block") {
      sub_block_idx = op.attrs(i).block_idx();
      break;
    }
  }
  Assert(sub_block_idx > 0, "Cannot find sub_block in while operator.");

  std::vector<std::shared_ptr<ONNX_NAMESPACE::NodeProto>> parameters;
  std::vector<std::string> input_names;
  std::vector<std::shared_ptr<ONNX_NAMESPACE::ValueInfoProto>> inputs;
  std::vector<std::string> output_names;
  std::vector<std::shared_ptr<ONNX_NAMESPACE::ValueInfoProto>> outputs;

  auto iter_name = MapperHelper::Get()->GenName("loop.iter");
  TensorInfo iter_info(
      iter_name, std::vector<int64_t>(1, 1), P2ODataType::INT64);
  inputs.push_back(std::move(MakeValueInfo(iter_info)));

  // Make cond
  input_names.push_back(cond_info[0].name);
  inputs.push_back(std::move(MakeValueInfo(cond_info[0])));
  outputs.push_back(std::move(std::move(MakeValueInfo(cond_info[0]))));

  // Make other inputs
  for (size_t i = 0; i < x_info.size(); ++i) {
    if (std::find(input_names.begin(), input_names.end(), x_info[i].name) !=
        input_names.end()) {
      continue;
    }

    if (!(x_info[i].is_tensor_array)) {
      // P2OLogger() << x_info[i].name << "is tensor array" << std::endl;
      inputs.push_back(std::move(MakeValueInfo(x_info[i])));
    }
    input_names.push_back(x_info[i].name);
    outputs.push_back(std::move(MakeValueInfo(x_info[i])));
  }

  graph = ExportBlock(
      parser, sub_block_idx, parameters, inputs, outputs, nullptr, true);

  /********************* Creat Body Gragh *********************/
  // Make Fake iter
  auto fake_iter = temp_helper->Constant(ONNX_NAMESPACE::TensorProto::INT64,
                                         std::vector<int64_t>(1, 1024));
  input_names.insert(input_names.begin(), fake_iter);
  for (int i = 2; i < input_names.size(); i++) {
    output_names.push_back(input_names[i]);
  }

  auto loop_node = temp_helper->MakeNode("Loop", input_names, output_names);
  AddAttribute(loop_node, "body", graph);
}
}  // namespace paddle2onnx
