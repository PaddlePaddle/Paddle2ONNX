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

#include "paddle2onnx/mapper/tensor/array_to_tensor.h"
#include <iostream>
#include <string>
#include <vector>

namespace paddle2onnx {
REGISTER_PIR_MAPPER(array_to_tensor, ArrayToTensorMapper)

int32_t ArrayToTensorMapper::GetMinOpsetVersion(bool verbose) {
  Logger(verbose, 17) << "ArrayToTensorMapper " << RequireOpset(17)
                      << std::endl;
  return 17;
}

void ArrayToTensorMapper::Opset17() {
  auto out_info = GetOutput("Out");
  auto out_index_info = GetOutput("OutIndex");
  std::string arr_name =
      pir_parser_->GetTensorArrayName(pir_op_idx_, if_in_cf_block);
  auto out_node =
      helper_->MakeNode("ConcatFromSequence", {arr_name}, {out_info[0].name});
  AddAttribute(out_node, "axis", axis_);
  AddAttribute(out_node, "new_axis", (int64_t)use_stack_);
  // ===== get out_index =====
  ONNX_NAMESPACE::GraphProto graph;
  graph.set_name("SequenceMap Graph in ArrayToTensor");
  // input
  auto input_value_info = std::make_shared<ONNX_NAMESPACE::ValueInfoProto>();
  input_value_info->set_name(arr_name);
  // TODO(wangmingkai02): why error occurred when set type to sequence_type?
  // auto type_proto =
  // input_value_info->mutable_type()->mutable_sequence_type()->mutable_elem_type();
  // type_proto->mutable_tensor_type()->set_elem_type(GetOnnxDtype(out_info[0].dtype));
  input_value_info->mutable_type()->mutable_tensor_type()->set_elem_type(
      GetOnnxDtype(out_info[0].dtype));
  *(graph.add_input()) = *(input_value_info.get());
  // output
  auto output_value_info = std::make_shared<ONNX_NAMESPACE::ValueInfoProto>();
  std::string seq_map_out_name =
      MapperHelper::Get()->GenName("sequence_map.out");
  output_value_info->set_name(seq_map_out_name);
  // auto out_type_proto =
  // output_value_info->mutable_type()->mutable_sequence_type()->mutable_elem_type();
  // out_type_proto->mutable_tensor_type()->set_elem_type(GetOnnxDtype(out_index_info[0].dtype));
  output_value_info->mutable_type()->mutable_tensor_type()->set_elem_type(
      GetOnnxDtype(out_index_info[0].dtype));
  *(graph.add_output()) = *(output_value_info.get());
  // nodes
  OnnxHelper temp_helper;
  auto shape_node = temp_helper.MakeNode("Shape", {arr_name});
  auto axis_node_name = temp_helper.Constant(ONNX_NAMESPACE::TensorProto::INT64,
                                             std::vector<int64_t>(1, axis_));
  auto gather_node =
      temp_helper.MakeNode("Gather", {shape_node->output(0), axis_node_name});
  AddAttribute(gather_node, "axis", (int64_t)0);
  auto cast_node = temp_helper.MakeNode(
      "Cast", {gather_node->output(0)}, {seq_map_out_name});
  AddAttribute(cast_node, "to", GetOnnxDtype(out_index_info[0].dtype));
  for (auto &item : temp_helper.nodes) {
    *(graph.add_node()) = (*item.get());
  }
  // ===== get out_index =====
  auto seq_map_node = helper_->MakeNode("SequenceMap", {arr_name});
  AddAttribute(seq_map_node, "body", graph);
  auto out_index_node = helper_->MakeNode("ConcatFromSequence",
                                          {seq_map_node->output(0)},
                                          {out_index_info[0].name});
  AddAttribute(out_index_node, "axis", (int64_t)0);
}

}  // namespace paddle2onnx
