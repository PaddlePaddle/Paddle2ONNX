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
void ModelExporter::ExportWhile(const PaddleParser& parser,
                                OnnxHelper* temp_helper, int32_t block_id,
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
  TensorInfo iter_info(iter_name, std::vector<int64_t>(1, 1),
                       P2ODataType::INT64);
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

  graph = ExportBlock(parser, sub_block_idx, parameters, inputs, outputs, nullptr, true);

  /********************* Creat Body Gragh *********************/
  // Make Fake iter
  auto fake_iter = temp_helper->Constant(ONNX_NAMESPACE::TensorProto::INT64,
                                         std::vector<int64_t>(1, 1024));
  input_names.insert(input_names.begin(), fake_iter);
  for(int i=2;i<input_names.size();i++) {
    output_names.push_back(input_names[i]);
  }
  
  auto loop_node = temp_helper->MakeNode("Loop", input_names, output_names);
  AddAttribute(loop_node, "body", graph);
}
}  // namespace paddle2onnx