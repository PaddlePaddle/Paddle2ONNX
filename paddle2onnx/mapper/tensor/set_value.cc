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

#include "paddle2onnx/mapper/tensor/set_value.h"

namespace paddle2onnx {
REGISTER_MAPPER(set_value, SetValueMapper)
REGISTER_PIR_MAPPER(set_value, SetValueMapper)
REGISTER_PIR_MAPPER(set_value_, SetValueMapper)
REGISTER_PIR_MAPPER(set_value_with_tensor, SetValueMapper)
REGISTER_PIR_MAPPER(set_value_with_tensor_, SetValueMapper)

int32_t SetValueMapper::GetMinOpsetVersion(bool verbose) {
  // TODO(wangmingkai02): update set_value mapper to support following features
  if (none_axes_.size() > 0) {
    Error() << "Attribute none_axes is not supported." << std::endl;
    return -1;
  }
  if (GetInput("Input")[0].dtype == P2ODataType::BOOL) {
    Error() << "Input X with data type of boolean is not supported."
            << std::endl;
    return -1;
  }
  return 17;
}

void SetValueMapper::Opset17() {
  auto input_info = GetInput("Input");
  auto output_info = GetOutput("Out");

  // Special case: if axes is empty, this is a full tensor assignment
  // Just copy the value to output (for set_value_with_tensor_ with empty axes)
  std::string op_type = OpType();
  bool is_set_value_with_tensor =
      (op_type.find("set_value_with_tensor") != std::string::npos);
  if (in_pir_mode && is_set_value_with_tensor && axes_.empty()) {
    auto value_info = GetInput(1);
    helper_->MakeNode("Identity", {value_info[0].name}, {output_info[0].name});
    return;
  }

  std::string starts = "";
  if (HasInput("StartsTensorList")) {
    // if negtive value exists, not supported
    starts = helper_->ConcatIndices(GetInput("StartsTensorList"));
  } else {
    starts = helper_->Constant(ONNX_NAMESPACE::TensorProto::INT64, starts_);
  }
  std::string ends = "";
  if (HasInput("EndsTensorList")) {
    ends = helper_->ConcatIndices(GetInput("EndsTensorList"));
  } else {
    // if out of range value in end exists, not supported
    ends = helper_->Constant(ONNX_NAMESPACE::TensorProto::INT64, ends_);
  }

  std::string steps = "";
  if (HasInput("StepsTensorList")) {
    steps = helper_->ConcatIndices(GetInput("StepsTensorList"));
  } else {
    steps = helper_->Constant(ONNX_NAMESPACE::TensorProto::INT64, steps_);
  }

  // process negative value in starts, ends
  int32_t input_rank = input_info[0].Rank();
  for (int32_t i = 0; i < axes_.size(); ++i) {
    if (axes_[i] < 0) {
      axes_[i] += input_rank;
    }
  }
  auto input_shape =
      helper_->MakeNode("Shape", {input_info[0].name})->output(0);
  auto zero = helper_->Constant(ONNX_NAMESPACE::TensorProto::INT64,
                                std::vector<int64_t>(axes_.size(), 0));
  auto starts_less_zero = helper_->MakeNode("Less", {starts, zero})->output(0);
  auto axes_const = helper_->Constant(std::vector<int64_t>(1, axes_.size()),
                                      ONNX_NAMESPACE::TensorProto::INT64,
                                      axes_);
  auto shape_bound = helper_->MakeNode("Gather", {input_shape, axes_const});
  AddAttribute(shape_bound, "axis", int64_t(0));
  auto add_start =
      helper_->MakeNode("Add", {starts, shape_bound->output(0)})->output(0);
  starts = helper_->MakeNode("Where", {starts_less_zero, add_start, starts})
               ->output(0);

  auto ends_less_zero = helper_->MakeNode("Less", {ends, zero})->output(0);
  auto add_end =
      helper_->MakeNode("Add", {ends, shape_bound->output(0)})->output(0);
  ends = helper_->MakeNode("Where", {ends_less_zero, add_end, ends})->output(0);
  ends = helper_->MakeNode("Min", {shape_bound->output(0), ends})->output(0);

  auto input_tensor = input_info[0].name;
  std::string value = "";
  int64_t value_rank = input_info[0].Rank();

  // Reuse op_type and is_set_value_with_tensor from earlier in function
  if (in_pir_mode && is_set_value_with_tensor) {
    // In PIR mode, set_value_with_tensor_ has value as second input (index 1)
    auto value_info = GetInput(1);
    value = value_info[0].name;
    value_rank = value_info[0].Rank();
  } else if (HasInput("ValueTensor")) {
    auto value_info = GetInput("ValueTensor");
    value = value_info[0].name;
    value_rank = value_info[0].Rank();
  } else if (HasInput("values")) {
    auto value_info = GetInput("values");
    value = value_info[0].name;
    value_rank = value_info[0].Rank();
  } else if (HasInput("value")) {
    // PIR mode: set_value_with_tensor_ uses "value" as input name
    auto value_info = GetInput("value");
    value = value_info[0].name;
    value_rank = value_info[0].Rank();
  } else {
    value_rank = shape_.size();
    int in_dtype = input_info[0].dtype;
    if (int_values_.size() > 0) {
      value = helper_->Assign(
          GetOnnxDtype(output_info[0].dtype), shape_, int_values_);
    } else if (fp32_values_.size() > 0) {
      value = helper_->Assign(
          GetOnnxDtype(output_info[0].dtype), shape_, fp32_values_);
    } else if (fp64_values_.size() > 0) {
      value = helper_->Assign(
          GetOnnxDtype(output_info[0].dtype), shape_, fp64_values_);
    }
  }

  if (axes_.size() > 1) {
    // process end <= start
    auto end_less_or_equal_start =
        helper_->MakeNode("LessOrEqual", {ends, starts})->output(0);
    auto cast_node = helper_->MakeNode("Cast", {end_less_or_equal_start});
    AddAttribute(cast_node, "to", ONNX_NAMESPACE::TensorProto::INT64);
    auto temp_cast = cast_node->output(0);
    auto temp_reduce_sum =
        helper_->MakeNode("ReduceSum", {temp_cast})->output(0);
    auto temp_cond = helper_->Flatten(temp_reduce_sum);
    auto cond_node = helper_->MakeNode("Cast", {temp_cond});
    AddAttribute(cond_node, "to", ONNX_NAMESPACE::TensorProto::BOOL);
    auto cond = cond_node->output(0);

    // ===== if then graph begin =====
    ONNX_NAMESPACE::GraphProto then_graph;
    then_graph.set_name("If then Graph in SetValue");

    // output
    auto then_output_info = std::make_shared<ONNX_NAMESPACE::ValueInfoProto>();
    then_output_info->set_name(output_info[0].name);
    then_output_info->mutable_type()->mutable_tensor_type()->set_elem_type(
        GetOnnxDtype(output_info[0].dtype));
    *(then_graph.add_output()) = *(then_output_info.get());

    // nodes
    OnnxHelper if_then_helper;
    if_then_helper.MakeNode("Identity", {input_tensor}, {output_info[0].name});
    for (auto &item : if_then_helper.nodes) {
      *(then_graph.add_node()) = (*item.get());
    }
    // ===== if then graph end =====

    // ===== if else graph begin =====
    ONNX_NAMESPACE::GraphProto else_graph;
    else_graph.set_name("If else Graph in SetValue");

    // output
    auto else_output_info = std::make_shared<ONNX_NAMESPACE::ValueInfoProto>();
    else_output_info->set_name(output_info[0].name);
    else_output_info->mutable_type()->mutable_tensor_type()->set_elem_type(
        GetOnnxDtype(output_info[0].dtype));
    *(else_graph.add_output()) = *(else_output_info.get());

    // nodes
    OnnxHelper if_else_helper;

    uint32_t complete_status = (1u << input_rank) - 1;
    uint32_t current_status = 0u;
    for (auto axis : axes_) {
      current_status |= (1u << axis);
    }
    std::vector<int64_t> other_axes;
    for (int32_t i = 0; i < input_rank; ++i) {
      if ((current_status & (1u << i)) == 0) {
        other_axes.push_back(i);
      }
    }
    auto other_axes_const =
        if_else_helper.Constant(std::vector<int64_t>(1, other_axes.size()),
                                ONNX_NAMESPACE::TensorProto::INT64,
                                other_axes);
    auto other_shape_bound =
        if_else_helper.MakeNode("Gather", {input_shape, other_axes_const});
    AddAttribute(other_shape_bound, "axis", int64_t(0));
    auto zero_starts = if_else_helper.Constant(
        ONNX_NAMESPACE::TensorProto::INT64,
        std::vector<int64_t>((input_rank - axes_.size()), 0));
    auto ones_steps = if_else_helper.Constant(
        ONNX_NAMESPACE::TensorProto::INT64,
        std::vector<int64_t>((input_rank - axes_.size()), 1));
    starts = if_else_helper.Concat({starts, zero_starts}, 0);
    ends = if_else_helper.Concat({ends, other_shape_bound->output(0)}, 0);
    steps = if_else_helper.Concat({steps, ones_steps}, 0);
    std::vector<int64_t> axes_tmp(axes_);
    axes_tmp.insert(axes_tmp.end(), other_axes.begin(), other_axes.end());
    std::vector<int64_t> axes_index;
    for (int32_t i = 0; i < input_rank; ++i) {
      for (int32_t j = 0; j < axes_tmp.size(); ++j) {
        if (axes_tmp[j] == i) {
          axes_index.push_back(j);
        }
      }
    }
    auto axes_index_const =
        if_else_helper.Constant(std::vector<int64_t>(1, axes_index.size()),
                                ONNX_NAMESPACE::TensorProto::INT64,
                                axes_index);
    auto sorted_starts_node =
        if_else_helper.MakeNode("Gather", {starts, axes_index_const});
    AddAttribute(sorted_starts_node, "axis", int64_t(0));
    auto sorted_ends_node =
        if_else_helper.MakeNode("Gather", {ends, axes_index_const});
    AddAttribute(sorted_ends_node, "axis", int64_t(0));
    auto sorted_steps_node =
        if_else_helper.MakeNode("Gather", {steps, axes_index_const});
    AddAttribute(sorted_steps_node, "axis", int64_t(0));

    auto ones = if_else_helper.Constant(ONNX_NAMESPACE::TensorProto::INT64,
                                        std::vector<int64_t>(input_rank, 1));
    auto start_seq =
        if_else_helper
            .MakeNode("SplitToSequence", {sorted_starts_node->output(0), ones})
            ->output(0);
    auto end_seq =
        if_else_helper
            .MakeNode("SplitToSequence", {sorted_ends_node->output(0), ones})
            ->output(0);
    auto step_seq =
        if_else_helper
            .MakeNode("SplitToSequence", {sorted_steps_node->output(0), ones})
            ->output(0);

    // ===== sequence map graph =====
    ONNX_NAMESPACE::GraphProto graph;
    graph.set_name("SequenceMap 1 Graph in SetValue");
    // input
    auto start_value_info = std::make_shared<ONNX_NAMESPACE::ValueInfoProto>();
    start_value_info->set_name(start_seq);
    start_value_info->mutable_type()->mutable_tensor_type()->set_elem_type(
        ONNX_NAMESPACE::TensorProto::INT64);
    *(graph.add_input()) = *(start_value_info.get());

    auto end_value_info = std::make_shared<ONNX_NAMESPACE::ValueInfoProto>();
    end_value_info->set_name(end_seq);
    end_value_info->mutable_type()->mutable_tensor_type()->set_elem_type(
        ONNX_NAMESPACE::TensorProto::INT64);
    *(graph.add_input()) = *(end_value_info.get());

    auto step_value_info = std::make_shared<ONNX_NAMESPACE::ValueInfoProto>();
    step_value_info->set_name(step_seq);
    step_value_info->mutable_type()->mutable_tensor_type()->set_elem_type(
        ONNX_NAMESPACE::TensorProto::INT64);
    *(graph.add_input()) = *(step_value_info.get());
    // output
    auto range_value_info = std::make_shared<ONNX_NAMESPACE::ValueInfoProto>();
    std::string range_out_name =
        MapperHelper::Get()->GenName("set_value.sequence_map.out.range");
    range_value_info->set_name(range_out_name);
    range_value_info->mutable_type()->mutable_tensor_type()->set_elem_type(
        ONNX_NAMESPACE::TensorProto::INT64);
    *(graph.add_output()) = *(range_value_info.get());

    auto size_value_info = std::make_shared<ONNX_NAMESPACE::ValueInfoProto>();
    std::string size_out_name =
        MapperHelper::Get()->GenName("set_value.sequence_map.out.size");
    size_value_info->set_name(size_out_name);
    size_value_info->mutable_type()->mutable_tensor_type()->set_elem_type(
        ONNX_NAMESPACE::TensorProto::INT64);
    *(graph.add_output()) = *(size_value_info.get());
    // nodes
    OnnxHelper temp_helper;
    temp_helper.MakeNode("Range",
                         {temp_helper.Squeeze(start_seq, {}),
                          temp_helper.Squeeze(end_seq, {}),
                          temp_helper.Squeeze(step_seq, {})},
                         {range_out_name});
    auto temp_scalar_size_name =
        temp_helper.MakeNode("Size", {range_out_name})->output(0);
    auto temp_zero = temp_helper.Constant(ONNX_NAMESPACE::TensorProto::INT64,
                                          std::vector<int64_t>(1, 0));
    auto temp_one = temp_helper.Constant(ONNX_NAMESPACE::TensorProto::INT64,
                                         std::vector<int64_t>(1, 1));
    auto temp_less_equal_zero =
        temp_helper.MakeNode("LessOrEqual", {temp_scalar_size_name, temp_zero})
            ->output(0);
    auto scalar_size_name =
        temp_helper
            .MakeNode("Where",
                      {temp_less_equal_zero, temp_one, temp_scalar_size_name})
            ->output(0);
    temp_helper.Unsqueeze(scalar_size_name, size_out_name, {0});  // wmk
    // auto range_2d = temp_helper.Unsqueeze(range, {1});
    // auto tile = temp_helper.MakeNode("Tile", {range_2d,
    // repeat_seq})->output(0); temp_helper.Reshape(tile, seq_map_out_name, {});
    for (auto &item : temp_helper.nodes) {
      *(graph.add_node()) = (*item.get());
    }

    // ===== sequence map graph =====
    auto seq_map_node_1 = if_else_helper.MakeNode(
        "SequenceMap", {start_seq, end_seq, step_seq}, 2);
    AddAttribute(seq_map_node_1, "body", graph);

    auto concat_size_node = if_else_helper.MakeNode(
        "ConcatFromSequence", {seq_map_node_1->output(1)});
    AddAttribute(concat_size_node, "axis", int64_t(0));
    auto concat_size = concat_size_node->output(0);
    // auto prod = if_else_helper.MakeNode("ReduceProd",
    // {concat_size})->output(0); // use default attrs
    auto zeros = if_else_helper.Constant(ONNX_NAMESPACE::TensorProto::INT64,
                                         std::vector<int64_t>(input_rank, 0));
    auto input_ranks =
        if_else_helper.Constant(ONNX_NAMESPACE::TensorProto::INT64,
                                std::vector<int64_t>(input_rank, input_rank));
    auto range_start_zero = if_else_helper.Constant(
        ONNX_NAMESPACE::TensorProto::INT64, std::vector<int64_t>(1, 0));
    auto range_end_input_rank =
        if_else_helper.Constant(ONNX_NAMESPACE::TensorProto::INT64,
                                std::vector<int64_t>(1, input_rank));
    auto range_delta = if_else_helper.Constant(
        ONNX_NAMESPACE::TensorProto::INT64, std::vector<int64_t>(1, 1));
    auto range1 =
        if_else_helper
            .MakeNode("Range",
                      {if_else_helper.Squeeze(range_start_zero, {}),
                       if_else_helper.Squeeze(range_end_input_rank, {}),
                       if_else_helper.Squeeze(range_delta, {})})
            ->output(0);
    auto range2 = if_else_helper.MakeNode("Add", {range1, ones})->output(0);
    auto slice_start_1 =
        if_else_helper.MakeNode("SplitToSequence", {zeros, ones})->output(0);
    auto slice_end_1 =
        if_else_helper.MakeNode("SplitToSequence", {range1, ones})->output(0);
    auto slice_start_2 =
        if_else_helper.MakeNode("SplitToSequence", {range2, ones})->output(0);
    auto slice_end_2 =
        if_else_helper.MakeNode("SplitToSequence", {input_ranks, ones})
            ->output(0);

    // ===== sequence map graph 2 =====
    ONNX_NAMESPACE::GraphProto graph2;
    graph2.set_name("SequenceMap 2 Graph in SetValue");
    // input

    auto slice_start_1_value_info =
        std::make_shared<ONNX_NAMESPACE::ValueInfoProto>();
    slice_start_1_value_info->set_name(slice_start_1);
    slice_start_1_value_info->mutable_type()
        ->mutable_tensor_type()
        ->set_elem_type(ONNX_NAMESPACE::TensorProto::INT64);
    *(graph2.add_input()) = *(slice_start_1_value_info.get());

    auto slice_end_1_value_info =
        std::make_shared<ONNX_NAMESPACE::ValueInfoProto>();
    slice_end_1_value_info->set_name(slice_end_1);
    slice_end_1_value_info->mutable_type()
        ->mutable_tensor_type()
        ->set_elem_type(ONNX_NAMESPACE::TensorProto::INT64);
    *(graph2.add_input()) = *(slice_end_1_value_info.get());

    auto slice_start_2_value_info =
        std::make_shared<ONNX_NAMESPACE::ValueInfoProto>();
    slice_start_2_value_info->set_name(slice_start_2);
    slice_start_2_value_info->mutable_type()
        ->mutable_tensor_type()
        ->set_elem_type(ONNX_NAMESPACE::TensorProto::INT64);
    *(graph2.add_input()) = *(slice_start_2_value_info.get());

    auto slice_end_2_value_info =
        std::make_shared<ONNX_NAMESPACE::ValueInfoProto>();
    slice_end_2_value_info->set_name(slice_end_2);
    slice_end_2_value_info->mutable_type()
        ->mutable_tensor_type()
        ->set_elem_type(ONNX_NAMESPACE::TensorProto::INT64);
    *(graph2.add_input()) = *(slice_end_2_value_info.get());

    auto range_out_value_info =
        std::make_shared<ONNX_NAMESPACE::ValueInfoProto>();
    range_out_value_info->set_name(seq_map_node_1->output(0));
    range_out_value_info->mutable_type()->mutable_tensor_type()->set_elem_type(
        ONNX_NAMESPACE::TensorProto::INT64);
    *(graph2.add_input()) = *(range_out_value_info.get());

    auto concat_size_value_info =
        std::make_shared<ONNX_NAMESPACE::ValueInfoProto>();
    concat_size_value_info->set_name(concat_size);
    concat_size_value_info->mutable_type()
        ->mutable_tensor_type()
        ->set_elem_type(ONNX_NAMESPACE::TensorProto::INT64);
    *(graph2.add_input()) = *(concat_size_value_info.get());

    // output
    auto index_value_info = std::make_shared<ONNX_NAMESPACE::ValueInfoProto>();
    std::string index_out_name =
        MapperHelper::Get()->GenName("set_value.sequence_map.out.index");
    index_value_info->set_name(index_out_name);
    index_value_info->mutable_type()->mutable_tensor_type()->set_elem_type(
        ONNX_NAMESPACE::TensorProto::INT64);
    *(graph2.add_output()) = *(index_value_info.get());

    // nodes
    OnnxHelper temp_helper2;
    auto slice_axes = temp_helper2.Constant(
        {1}, ONNX_NAMESPACE::TensorProto::INT64, static_cast<int64_t>(0));
    auto prefix_size =
        temp_helper2
            .MakeNode("Slice",
                      {concat_size, slice_start_1, slice_end_1, slice_axes})
            ->output(0);
    auto suffix_size =
        temp_helper2
            .MakeNode("Slice",
                      {concat_size, slice_start_2, slice_end_2, slice_axes})
            ->output(0);
    auto temp_repeat_1 = temp_helper2.MakeNode("ReduceProd", {prefix_size})
                             ->output(0);  // use default attrs
    auto temp_repeat_2 =
        temp_helper2.MakeNode("ReduceProd", {suffix_size})->output(0);
    auto repeat_1 = temp_helper2.Reshape(temp_repeat_1, {1});
    auto repeat_2 = temp_helper2.Reshape(temp_repeat_2, {1});
    auto repeats = temp_helper2.Concat({repeat_1, repeat_2}, 0);
    auto range_2d = temp_helper2.Unsqueeze(seq_map_node_1->output(0), {1});
    auto tile = temp_helper2.MakeNode("Tile", {range_2d, repeats})->output(0);
    auto tmep_prod =
        temp_helper2.MakeNode("ReduceProd", {concat_size})->output(0);
    auto prod = temp_helper2.Reshape(tmep_prod, {1});
    auto reshape = temp_helper2.MakeNode("Reshape", {tile, prod})->output(0);
    temp_helper2.Unsqueeze(reshape, index_out_name, {1});
    for (auto &item : temp_helper2.nodes) {
      *(graph2.add_node()) = (*item.get());
    }
    // ===== sequence map graph 2 =====

    auto seq_map_node_2 = if_else_helper.MakeNode("SequenceMap",
                                                  {slice_start_1,
                                                   slice_end_1,
                                                   slice_start_2,
                                                   slice_end_2,
                                                   seq_map_node_1->output(0),
                                                   concat_size},
                                                  1);
    AddAttribute(seq_map_node_2, "body", graph2);
    auto indices_node = if_else_helper.MakeNode("ConcatFromSequence",
                                                {seq_map_node_2->output(0)});
    AddAttribute(indices_node, "axis", int64_t(1));
    auto indices = indices_node->output(0);

    // construct update
    auto flatten_value = if_else_helper.Flatten(value);
    auto update_shape = if_else_helper.Flatten(
        if_else_helper.MakeNode("ReduceProd", {concat_size})->output(0));
    auto expand_value =
        if_else_helper.MakeNode("Expand", {flatten_value, update_shape})
            ->output(0);
    if_else_helper.MakeNode("ScatterND",
                            {input_tensor, indices, expand_value},
                            {output_info[0].name});

    for (auto &item : if_else_helper.nodes) {
      *(else_graph.add_node()) = (*item.get());
    }
    // ===== if else graph end =====
    auto if_node = helper_->MakeNode("If", {cond}, {output_info[0].name});
    AddAttribute(if_node, "else_branch", else_graph);
    AddAttribute(if_node, "then_branch", then_graph);
  } else {
    std::string axes = helper_->Constant(
        {1}, ONNX_NAMESPACE::TensorProto::INT64, int64_t(axes_[0]));

    auto sliced_data =
        helper_->MakeNode("Slice", {input_tensor, starts, ends, axes, steps})
            ->output(0);

    auto sliced_shape = helper_->MakeNode("Shape", {sliced_data})->output(0);
    if (decrease_axes_.size() > 0 && value_rank != input_info[0].Rank()) {
      value = helper_->Unsqueeze(value, decrease_axes_);
    }
    auto expand_value =
        helper_->MakeNode("Expand", {value, sliced_shape})->output(0);

    // Range 的输入要求时Scalar，当axes_ > 1时就不满足了
    auto indices = helper_
                       ->MakeNode("Range",
                                  {helper_->Squeeze(starts, {}),
                                   helper_->Squeeze(ends, {}),
                                   helper_->Squeeze(steps, {})})
                       ->output(0);
    if (axes_[0] == 0) {
      indices = helper_->Unsqueeze(indices, {1});
      helper_->MakeNode("ScatterND",
                        {input_tensor, indices, expand_value},
                        {output_info[0].name});
    } else {
      std::vector<int64_t> indices_shape(input_info[0].Rank(), 1);
      indices_shape[axes_[0]] = -1;
      indices = helper_->Reshape(indices, indices_shape);
      auto one = helper_->Constant(
          {1}, ONNX_NAMESPACE::TensorProto::INT64, int64_t(1));
      if (axes_[0] == input_info[0].Rank() - 1) {
        auto part_shape = helper_->Slice(sliced_shape, {0}, {0}, {axes_[0]});
        auto tiled_shape = helper_->Concat({part_shape, one}, 0);
        indices = helper_->MakeNode("Tile", {indices, tiled_shape})->output(0);
      } else {
        auto part_0_shape = helper_->Slice(sliced_shape, {0}, {0}, {axes_[0]});
        auto part_1_shape = helper_->Slice(
            sliced_shape, {0}, {axes_[0] + 1}, {input_info[0].Rank()});
        auto tiled_shape =
            helper_->Concat({part_0_shape, one, part_1_shape}, 0);
        indices = helper_->MakeNode("Tile", {indices, tiled_shape})->output(0);
      }
      auto scatter_node =
          helper_->MakeNode("ScatterElements",
                            {input_tensor, indices, expand_value},
                            {output_info[0].name});
      AddAttribute(scatter_node, "axis", axes_[0]);
    }
  }
}

}  // namespace paddle2onnx
