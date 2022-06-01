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

#include "paddle2onnx/mapper/quantize_helper.h"

namespace paddle2onnx {

void QuantizeModelProcess::remove_node_by_name(
    const std::map<std::string,
                   std::vector<std::shared_ptr<ONNX_NAMESPACE::NodeProto>>>&
        name2node_dict,
    std::vector<std::shared_ptr<ONNX_NAMESPACE::NodeProto>>* nodes,
    const std::string& name) {
  for (auto iter = nodes->begin(); iter != nodes->end(); iter++) {
    if ((*iter)->name() == name) {
      std::string input_name = (*iter)->input(0);
      std::string output_name = (*iter)->output(0);
      nodes->erase(iter);
      replace_input_of_all_nodes(name2node_dict, output_name, input_name);
      return;
    }
  }
}

void QuantizeModelProcess::replace_input_of_all_nodes(
    const std::map<std::string,
                   std::vector<std::shared_ptr<ONNX_NAMESPACE::NodeProto>>>&
        name2node_dict,
    const std::string& old_name, const std::string& new_name) {
  auto iter = name2node_dict.find(old_name);
  std::vector<std::shared_ptr<ONNX_NAMESPACE::NodeProto>> need_remove_nodes;
  if (iter != name2node_dict.end()) {
    need_remove_nodes = iter->second;
  }
  for (auto& node : need_remove_nodes) {
    std::vector<std::string> inputs;
    for (size_t i = 0; i < node->input_size(); ++i) {
      if (node->input(i) == old_name) {
        inputs.push_back(new_name);
      } else {
        inputs.push_back(node->input(i));
      }
    }
    node->clear_input();
    for (auto in : inputs) {
      node->add_input(in);
    }
  }
}

void QuantizeModelProcess::input_name_to_nodes(
    const std::vector<std::shared_ptr<ONNX_NAMESPACE::NodeProto>>& nodes,
    std::map<std::string,
             std::vector<std::shared_ptr<ONNX_NAMESPACE::NodeProto>>>*
        name2node_dict) {
  for (auto& node : nodes) {
    for (size_t i = 0; i < node->input_size(); ++i) {
      std::string node_input = node->input(i);
      if (name2node_dict->find(node_input) != name2node_dict->end()) {
        (*name2node_dict)[node_input].push_back(node);
      } else {
        std::vector<std::shared_ptr<ONNX_NAMESPACE::NodeProto>> next_nodes;
        (*name2node_dict)[node_input] = next_nodes;
        (*name2node_dict)[node_input].push_back(node);
      }
    }
  }
}

void QuantizeModelProcess::process_quantize_model(
    std::vector<std::shared_ptr<ONNX_NAMESPACE::NodeProto>>* parameters,
    std::vector<std::shared_ptr<ONNX_NAMESPACE::ValueInfoProto>>* inputs,
    std::vector<std::shared_ptr<ONNX_NAMESPACE::ValueInfoProto>>* outputs,
    std::vector<std::shared_ptr<ONNX_NAMESPACE::NodeProto>>* nodes,
    OnnxHelper& helper, const std::string deploy_backend) {
  if (deploy_backend == "others") {
    remove_all_quantize_ops(parameters, inputs, outputs, nodes, helper);
    std::ofstream outfile;
    outfile.open("max_range.txt", std::ios::out);
    for (auto iter = helper.quantize_info.begin();
         iter != helper.quantize_info.end(); iter++) {
      std::string log = iter->first;
      auto scale = iter->second.scale_;
      if (scale.size() == 1) {
        log = log + ": " + std::to_string(scale[0] * 127);
      }
      outfile << log << std::endl;
    }
    outfile.close();
  }
}

void QuantizeModelProcess::remove_all_quantize_ops(
    std::vector<std::shared_ptr<ONNX_NAMESPACE::NodeProto>>* parameters,
    std::vector<std::shared_ptr<ONNX_NAMESPACE::ValueInfoProto>>* inputs,
    std::vector<std::shared_ptr<ONNX_NAMESPACE::ValueInfoProto>>* outputs,
    std::vector<std::shared_ptr<ONNX_NAMESPACE::NodeProto>>* nodes,
    OnnxHelper& helper) {
  int node_num = 0;
  for (auto iter = helper.quantize_info.begin();
       iter != helper.quantize_info.end(); iter++) {
    node_num++;
  }
  std::map<std::string, std::vector<std::shared_ptr<ONNX_NAMESPACE::NodeProto>>>
      name2node_dict;
  input_name_to_nodes(*nodes, &name2node_dict);
  for (auto iter = nodes->begin(); iter < nodes->end(); iter++) {
    auto node = *iter;
    if (node->op_type() != "QuantizeLinear") {
      continue;
    }
    std::string input_name = node->input(0);
    remove_node_by_name(name2node_dict, nodes, node->name());
    auto next_node_names = name2node_dict[node->output(0)];
    std::string output_name = next_node_names[0]->output(0);
    remove_node_by_name(name2node_dict, nodes, next_node_names[0]->name());
    replace_input_of_all_nodes(name2node_dict, output_name, input_name);
  }
}
}
