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

#include <unordered_set>

namespace paddle2onnx {
void ModelExporter::ExportWhile(const PaddlePirParser& pir_parser,
                                OnnxHelper* temp_helper,
                                pir::Operation* op) {
  // ================================
  //  construct loop body sub graph
  // ================================
  std::vector<TensorInfo> inputs_info;
  std::vector<TensorInfo> outputs_info;
  std::vector<std::shared_ptr<ONNX_NAMESPACE::NodeProto>> extra_nodes;

  // For standalone PIR, "while" op has:
  //   operand[0] = cond
  //   operand[1+] = loop vars
  //   region[0].block[0] = body block
  //     block.args[0+] = block arguments (corresponding to operands[1+])
  auto cond_value = op->num_operands() > 0 ? op->operand(0).source() : pir::Value();
  auto cond_info = cond_value.defining_op()
      ? pir_parser.GetTensorInfo(cond_value)
      : std::vector<TensorInfo>();

  std::unordered_set<std::string> names;
  for (size_t index = 1; index < op->num_operands(); index++) {
    const pir::Value& value = op->operand(index).source();
    std::string name = pir_parser.GetSubBlockOpOutputName(value);
    if (names.count(name)) {
      // there are duplicated variable names in while op's operands.
      name = temp_helper->MakeNode("Identity", {name})->output(0);
      pir_parser.while_op_args_name_map[value.id()] = name;
    } else {
      names.insert(name);
    }
    inputs_info.push_back(pir_parser.GetTensorInfo(name, value.type()));
  }
  pir_parser.GetWhileInputValuesAndArgsMappings(op);

  std::vector<pir::Operation*> sub_blocks_ops_copy(pir_parser.sub_blocks_ops);
  pir_parser.sub_blocks_ops.clear();

  // The body block is in region[0], block[0]
  if (op->num_regions() > 0) {
    auto& body_block = op->region(0);
    for (auto& body_op : body_block.ops()) {
      if (body_op.name() != "builtin.parameter") {
        pir_parser.sub_blocks_ops.push_back(&body_op);
      }
    }
  }

  if (!pir_parser.sub_blocks_ops.empty()) {
    // get cf.yield op input (last op in sub-block)
    pir::Operation* cf_yield_op = pir_parser.sub_blocks_ops.back();
    if (cf_yield_op->name() != "cf.yield") {
      throw std::runtime_error(
          "The last op of a control flow sub-block must be cf.yield, but got " +
          cf_yield_op->name());
    }
    for (size_t oi = 0; oi < cf_yield_op->num_operands(); oi++) {
      pir::Value value = cf_yield_op->operand(oi).source();
      auto* def_op = value.defining_op();
      if (def_op && def_op->GetParent() != cf_yield_op->GetParent()) {
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
    throw std::runtime_error(
        "The number of ops of a control flow sub-block cannot be zero.");
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
  if (!cond_info.empty())
    inputs.push_back(std::move(MakeValueInfo(cond_info[0])));
  for (size_t i = 0; i < inputs_info.size(); ++i) {
    inputs.push_back(std::move(MakeValueInfo(inputs_info[i])));
  }
  // outputs
  for (size_t i = 0; i < outputs_info.size(); ++i) {
    outputs.push_back(std::move(MakeValueInfo(outputs_info[i])));
  }

  // Export the body block
  if (op->num_regions() > 0) {
    auto& body_block = op->region(0);
    // We need to pass a Block* to ExportBlock
    graph = ExportBlock(pir_parser, &body_block, &parameters, &inputs, &outputs, true, true);
  }

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
  if (!cond_info.empty())
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
      inputs.push_back(std::move(MakeValueInfo(x_info[i])));
    }
    input_names.push_back(x_info[i].name);
    outputs.push_back(std::move(MakeValueInfo(x_info[i])));
  }

  graph = ExportBlock(
      parser, sub_block_idx, &parameters, &inputs, &outputs, nullptr, true);

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
