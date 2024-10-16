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
namespace paddle2onnx {
ONNX_NAMESPACE::GraphProto ModelExporter::ExportFillConstant(
    const PaddleParser &parser, OnnxHelper *temp_helper, int32_t block_id,
    int32_t op_id, const std::string &output_name) {
  ONNX_NAMESPACE::GraphProto graph;
  graph.set_name("PaddlePaddle fill_constant Graph " + std::to_string(op_id));

  // Add input

  // Add node
  auto &nodes = temp_helper->nodes;
  for (int i = 0; i < nodes.size(); i++) {
    auto &item = nodes[i];
    if (item->output(0) == output_name) {
      *(graph.add_node()) = (*item.get());
      nodes.erase(nodes.begin() + i);
      break;
    }
  }

  // Add output
  auto out_info = parser.GetOpOutput(block_id, op_id, "Out");
  *(graph.add_output()) = (*MakeValueInfo(out_info[0]));
  return std::move(graph);
}

ONNX_NAMESPACE::GraphProto ModelExporter::ExportConditionalBlock(
    const PaddleParser &parser, OnnxHelper *temp_helper, int32_t block_id,
    int32_t op_id, const std::string &output_name) {
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

  // Export sub_block
  std::vector<std::shared_ptr<ONNX_NAMESPACE::NodeProto>> temp_parameters;
  std::vector<std::shared_ptr<ONNX_NAMESPACE::ValueInfoProto>> temp_inputs;
  std::vector<std::shared_ptr<ONNX_NAMESPACE::ValueInfoProto>> temp_outputs;
  auto out_info = parser.GetOpOutput(block_id, op_id, "Out");
  for (int index = 0; index < out_info.size(); index++) {
    if (out_info[index].name != output_name) {
      continue;
    }
    temp_outputs.push_back(std::move(MakeValueInfo(out_info[index])));
  }

  return ExportBlock(parser, sub_block_idx, temp_parameters, temp_inputs,
                     temp_outputs);
}

void ModelExporter::ExportSelectInput(const PaddleParser &parser,
                                      OnnxHelper *temp_helper, int32_t block_id,
                                      int32_t op_id) {
  auto input_info = parser.GetOpInput(block_id, op_id, "X");

  Assert(input_info.size() == 2,
         "Only support when number of select_input's input_node is 2.");

  ONNX_NAMESPACE::GraphProto graphs[2];
  for (int i = 0; i < input_info.size(); i++) {
    auto node_name = input_info[i].name;
    auto conditional_block_cood_it = sub_block_map_.find(node_name);
    Assert(conditional_block_cood_it != sub_block_map_.end(),
           "Can't find select_input else_input node.");
    auto conditional_block_cood = conditional_block_cood_it->second;
    auto node = parser.GetOpDesc(conditional_block_cood.first,
                                 conditional_block_cood.second);

    if (node.type().find("conditional_block") != std::string::npos) {
      graphs[i] = ExportConditionalBlock(
          parser, temp_helper, conditional_block_cood.first,
          conditional_block_cood.second, node_name);
    } else {
      graphs[i] =
          ExportFillConstant(parser, temp_helper, conditional_block_cood.first,
                             conditional_block_cood.second, node_name);
    }
  }

  auto cond_info = parser.GetOpInput(block_id, op_id, "Mask");
  auto output_info = parser.GetOpOutput(block_id, op_id, "Out");
  auto cond_name = temp_helper->AutoCast(cond_info[0].name, cond_info[0].dtype,
                                         P2ODataType::BOOL);
  auto node = temp_helper->MakeNode("If", {cond_name}, {output_info[0].name});
  AddAttribute(node, "else_branch", graphs[0]);
  AddAttribute(node, "then_branch", graphs[1]);
}
}  // namespace paddle2onnx