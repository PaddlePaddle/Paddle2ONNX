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

#include "paddle2onnx/mapper/onnx_helper.h"

namespace paddle2onnx {

void AddAttribute(std::shared_ptr<ONNX_NAMESPACE::NodeProto> node,
                  const std::string& name, const int64_t& value) {
  auto attr = node->add_attribute();
  attr->set_name(name);
  attr->set_i(value);
  attr->set_type(ONNX_NAMESPACE::AttributeProto::INT);
}

void AddAttribute(std::shared_ptr<ONNX_NAMESPACE::NodeProto> node,
                  const std::string& name, const float& value) {
  auto attr = node->add_attribute();
  attr->set_name(name);
  attr->set_f(value);
  attr->set_type(ONNX_NAMESPACE::AttributeProto::FLOAT);
}

void AddAttribute(std::shared_ptr<ONNX_NAMESPACE::NodeProto> node,
                  const std::string& name, const std::string& value) {
  auto attr = node->add_attribute();
  attr->set_name(name);
  attr->set_s(value);
  attr->set_type(ONNX_NAMESPACE::AttributeProto::STRING);
}

void AddAttribute(std::shared_ptr<ONNX_NAMESPACE::NodeProto> node,
                  const std::string& name, const std::vector<int64_t>& values) {
  auto attr = node->add_attribute();
  attr->set_name(name);
  for (auto& item : values) {
    attr->add_ints(item);
  }
  attr->set_type(ONNX_NAMESPACE::AttributeProto::INTS);
}

void AddAttribute(std::shared_ptr<ONNX_NAMESPACE::NodeProto> node,
                  const std::string& name, const std::vector<float>& values) {
  auto attr = node->add_attribute();
  attr->set_name(name);
  for (auto& item : values) {
    attr->add_floats(item);
  }
  attr->set_type(ONNX_NAMESPACE::AttributeProto::FLOAT);
}

void AddAttribute(std::shared_ptr<ONNX_NAMESPACE::NodeProto> node,
                  const std::string& name,
                  ONNX_NAMESPACE::TensorProto_DataType dtype) {
  auto attr = node->add_attribute();
  attr->set_name(name);
  attr->set_i(static_cast<int>(dtype));
  attr->set_type(ONNX_NAMESPACE::AttributeProto::INT);
}

ONNX_NAMESPACE::TensorProto_DataType GetOnnxDtype(int32_t paddle_dtype) {
  Assert(paddle_dtype >= 0 && paddle_dtype <= 6, "Unknow data type of weight.");
  auto onnx_dtype = ONNX_NAMESPACE::TensorProto::FLOAT;
  if (paddle_dtype == P2ODataType::BOOL) {
    onnx_dtype = ONNX_NAMESPACE::TensorProto::BOOL;
  } else if (paddle_dtype == P2ODataType::INT16) {
    onnx_dtype = ONNX_NAMESPACE::TensorProto::INT16;
  } else if (paddle_dtype == P2ODataType::INT32) {
    onnx_dtype = ONNX_NAMESPACE::TensorProto::INT32;
  } else if (paddle_dtype == P2ODataType::INT64) {
    onnx_dtype = ONNX_NAMESPACE::TensorProto::INT64;
  } else if (paddle_dtype == P2ODataType::FP16) {
    onnx_dtype = ONNX_NAMESPACE::TensorProto::FLOAT16;
  } else if (paddle_dtype == P2ODataType::FP32) {
    onnx_dtype = ONNX_NAMESPACE::TensorProto::FLOAT;
  } else {
    onnx_dtype = ONNX_NAMESPACE::TensorProto::DOUBLE;
  }
  return onnx_dtype;
}

std::shared_ptr<ONNX_NAMESPACE::NodeProto> MakeConstant(const std::string& name,
                                                        const Weight& weight) {
  auto node = std::make_shared<ONNX_NAMESPACE::NodeProto>();
  node->set_op_type("Constant");
  node->add_output(name);
  auto attr = node->add_attribute();
  attr->set_name("value");
  attr->set_type(ONNX_NAMESPACE::AttributeProto::TENSOR);
  auto tensor = attr->mutable_t();
  tensor->set_name(name);
  auto onnx_dtype = GetOnnxDtype(weight.dtype);
  tensor->set_data_type(onnx_dtype);
  for (auto& dim : weight.shape) {
    tensor->add_dims(dim);
  }
  tensor->set_raw_data(std::string(weight.buffer.data(), weight.buffer.size()));
  return node;
}

std::shared_ptr<ONNX_NAMESPACE::NodeProto> MakeConstant(const Weight& weight) {
  auto node_name = MapperHelper::Get()->GenName("auto.constant");
  return MakeConstant(node_name, weight);
}

std::shared_ptr<ONNX_NAMESPACE::ValueInfoProto> MakeValueInfo(
    const TensorInfo& info) {
  auto value_info = std::make_shared<ONNX_NAMESPACE::ValueInfoProto>();
  value_info->set_name(info.name);
  auto type_proto = value_info->mutable_type();
  auto tensor_type_proto = type_proto->mutable_tensor_type();
  tensor_type_proto->set_elem_type(GetOnnxDtype(info.dtype));
  auto shape = tensor_type_proto->mutable_shape();
  for (auto& dim : info.shape) {
    shape->add_dim()->set_dim_value(dim);
  }
  return value_info;
}

std::shared_ptr<ONNX_NAMESPACE::NodeProto> OnnxHelper::MakeNode(
    const std::string& op_type, const std::vector<std::string>& inputs,
    const std::vector<std::string>& outputs) {
  auto node = std::make_shared<ONNX_NAMESPACE::NodeProto>();
  auto node_name = MapperHelper::Get()->GenName(op_type);
  node->set_name(node_name);
  node->set_op_type(op_type);
  for (size_t i = 0; i < inputs.size(); ++i) {
    node->add_input(inputs[i]);
  }
  for (size_t i = 0; i < outputs.size(); ++i) {
    node->add_output(outputs[i]);
  }
  nodes.push_back(node);
  return node;
}

std::shared_ptr<ONNX_NAMESPACE::NodeProto> OnnxHelper::MakeNode(
    const std::string& op_type, const std::vector<std::string>& inputs,
    int num_outputs) {
  auto node = std::make_shared<ONNX_NAMESPACE::NodeProto>();
  auto node_name = MapperHelper::Get()->GenName(op_type);
  node->set_name(node_name);
  node->set_op_type(op_type);
  for (size_t i = 0; i < inputs.size(); ++i) {
    node->add_input(inputs[i]);
  }
  std::vector<std::string> outputs;
  for (auto i = 0; i < num_outputs; ++i) {
    outputs.push_back(MapperHelper::Get()->GenName(op_type));
  }
  for (size_t i = 0; i < outputs.size(); ++i) {
    node->add_output(outputs[i]);
  }
  nodes.push_back(node);
  return node;
}

std::string OnnxHelper::AutoCast(const std::string& input,
                                 int32_t input_paddle_dtype,
                                 int32_t to_paddle_dtype) {
  std::string output = input;
  if (input_paddle_dtype == to_paddle_dtype) {
    return output;
  }
  output = MapperHelper::Get()->GenName("auto.cast");
  auto cast_node = MakeNode("Cast", {input}, {output});
  AddAttribute(cast_node, "to", GetOnnxDtype(to_paddle_dtype));
  return cast_node->output(0);
}

std::string OnnxHelper::AutoCast(const std::string& input,
                                 const std::string& output,
                                 int32_t input_paddle_dtype,
                                 int32_t to_paddle_dtype) {
  if (input_paddle_dtype == to_paddle_dtype) {
    return input;
  }
  auto cast_node = MakeNode("Cast", {input}, {output});
  AddAttribute(cast_node, "to", GetOnnxDtype(to_paddle_dtype));
  return cast_node->output(0);
}

std::shared_ptr<ONNX_NAMESPACE::NodeProto> OnnxHelper::MakeConstant(
    const std::shared_ptr<ONNX_NAMESPACE::NodeProto>& node) {
  nodes.push_back(node);
  return node;
}

std::string OnnxHelper::Clip(const std::string& input,
                             const std::string& output, const float& min,
                             const float& max, const int32_t& in_dtype) {
  std::string input_name;
  if (in_dtype == P2ODataType::FP64) {
    input_name = AutoCast(input, P2ODataType::FP64, P2ODataType::FP32);
  } else {
    input_name = input;
  }
  if (opset_version < 11) {
    if (in_dtype == P2ODataType::FP64) {
      auto node = MakeNode("Clip", {input_name});
      AddAttribute(node, "max", max);
      AddAttribute(node, "min", min);
      auto res = AutoCast(node->output(0), output, P2ODataType::FP32,
                          P2ODataType::FP64);
      return res;
    } else {
      auto node = MakeNode("Clip", {input_name}, {output});
      AddAttribute(node, "max", max);

      AddAttribute(node, "min", min);

      return node->output(0);
    }
  } else {
    if (in_dtype == P2ODataType::FP64) {
      std::string min_name;
      int32_t dtype = P2ODataType::FP32;
      min_name = MakeConstant({1}, GetOnnxDtype(dtype), min)->output(0);
      std::string max_name;
      max_name = MakeConstant({1}, GetOnnxDtype(dtype), max)->output(0);
      auto node = MakeNode("Clip", {input_name, min_name, max_name});
      auto res = AutoCast(node->output(0), {output}, P2ODataType::FP32,
                          P2ODataType::FP64);
      return res;
    } else {
      std::string min_name;
      int32_t dtype = in_dtype;
      min_name = MakeConstant({1}, GetOnnxDtype(dtype), min)->output(0);
      std::string max_name;
      max_name = MakeConstant({1}, GetOnnxDtype(dtype), max)->output(0);
      auto node = MakeNode("Clip", {input_name, min_name, max_name}, {output});
      return node->output(0);
    }
  }
}

std::string OnnxHelper::Clip(const std::string& input, const float& min,
                             const float& max, const int32_t& in_dtype) {
  std::string output = MapperHelper::Get()->GenName("helper.clip");
  return Clip(input, output, min, max, in_dtype);
}

std::string OnnxHelper::Squeeze(const std::string& input,
                                const std::string& output,
                                const std::vector<int64_t>& axes) {
  if (axes.size() == 0) {
    auto node = MakeNode("Squeeze", {input}, {output});
  } else {
    if (opset_version < 13) {
      auto node = MakeNode("Squeeze", {input}, {output});
      AddAttribute(node, "axes", axes);
    } else {
      auto axes_node = MakeConstant(ONNX_NAMESPACE::TensorProto::INT64, axes);
      auto node = MakeNode("Squeeze", {input, axes_node->output(0)}, {output});
    }
  }
  return output;
}

std::string OnnxHelper::Squeeze(const std::string& input,
                                const std::vector<int64_t>& axes) {
  std::string output = MapperHelper::Get()->GenName("helper.squeeze");
  return Squeeze(input, output, axes);
}

std::string OnnxHelper::Reshape(const std::string& input,
                                const std::string& output,
                                const std::vector<int64_t>& shape) {
  if (opset_version < 6) {
    auto node = MakeNode("Reshape", {input}, {output});
    AddAttribute(node, "shape", shape);
  } else {
    auto shape_node = MakeConstant(ONNX_NAMESPACE::TensorProto::INT64, shape);
    auto node = MakeNode("Reshape", {input, shape_node->output(0)}, {output});
  }
  return output;
}

std::string OnnxHelper::Reshape(const std::string& input,
                                const std::vector<int64_t>& shape) {
  std::string output = MapperHelper::Get()->GenName("helper.reshape");
  return Reshape(input, output, shape);
}

std::string OnnxHelper::Flatten(const std::string& input,
                                const std::string& output) {
  return Reshape(input, output, std::vector<int64_t>(1, -1));
}

std::string OnnxHelper::Flatten(const std::string& input) {
  std::string output = MapperHelper::Get()->GenName("helper.flatten");
  return Flatten(input, output);
}

std::string OnnxHelper::Slice(const std::string& input,
                              const std::string& output,
                              const std::vector<int64_t>& axes,
                              const std::vector<int64_t>& starts,
                              const std::vector<int64_t>& ends) {
  if (opset_version < 10) {
    auto node = MakeNode("Slice", {input}, {output});
    AddAttribute(node, "axes", axes);
    AddAttribute(node, "starts", starts);
    AddAttribute(node, "ends", ends);
  } else {
    auto axes_node = MakeConstant(ONNX_NAMESPACE::TensorProto::INT64, axes);
    auto starts_node = MakeConstant(ONNX_NAMESPACE::TensorProto::INT64, starts);
    auto ends_node = MakeConstant(ONNX_NAMESPACE::TensorProto::INT64, ends);
    auto node = MakeNode("Slice", {input, starts_node->output(0),
                                   ends_node->output(0), axes_node->output(0)},
                         {output});
  }
  return output;
}

std::string OnnxHelper::Slice(const std::string& input,
                              const std::vector<int64_t>& axes,
                              const std::vector<int64_t>& starts,
                              const std::vector<int64_t>& ends) {
  std::string output = MapperHelper::Get()->GenName("helper.slice");
  return Slice(input, output, axes, starts, ends);
}

std::vector<std::string> OnnxHelper::Split(
    const std::string& input, const std::vector<std::string>& outputs,
    const std::vector<int64_t>& split, int64_t axis) {
  Assert(outputs.size() > 0 || split.size() > 0,
         "OnnxHelper::Split requires the size of outputs or the size of split "
         "> 0.");
  auto node = std::make_shared<ONNX_NAMESPACE::NodeProto>();
  auto node_name = MapperHelper::Get()->GenName("Split");
  node->set_name(node_name);
  node->set_op_type("Split");
  node->add_input(input);
  for (size_t i = 0; i < outputs.size(); ++i) {
    node->add_output(outputs[i]);
  }
  AddAttribute(node, "axis", axis);
  if (split.size() > 0) {
    Assert(outputs.size() == split.size(),
           "OnnxHelper::Split While size of outputs and the size of split both "
           "> 0, their size must be same.");
    if (opset_version < 13) {
      AddAttribute(node, "split", split);
    } else {
      auto split_const = Constant(ONNX_NAMESPACE::TensorProto::INT64, split);
      node->add_input(split_const);
    }
  }
  nodes.push_back(node);
  return outputs;
}

std::vector<std::string> OnnxHelper::Split(const std::string& input,
                                           const std::vector<int64_t>& split,
                                           int64_t axis) {
  Assert(split.size() > 0,
         "OnnxHelper::Split requires the size of parameter split > 0.");
  std::vector<std::string> outputs(split.size());
  for (size_t i = 0; i < split.size(); ++i) {
    outputs[i] = MapperHelper::Get()->GenName("helper.split");
  }
  return Split(input, outputs, split, axis);
}

std::vector<std::string> OnnxHelper::DtypeAlignment(
    const std::vector<TensorInfo>& input_info, int32_t& out_dtype) {
  Assert(input_info.size() > 0,
         "OnnxHelper::DtypeAlignment requires the size of input info > 0.");
  std::vector<int32_t> input_dtypes;
  input_dtypes.reserve(input_info.size());
  for (auto i = 0; i < input_info.size(); i++) {
    input_dtypes.push_back(input_info[i].dtype);
  }
  int32_t max_index = -1;
  for (auto i : input_dtypes) {
    if (i > max_index) max_index = i;
  }
  out_dtype = max_index;
  std::vector<std::string> casted_node;
  casted_node.reserve(input_info.size());
  for (auto i = 0; i < input_info.size(); i++) {
    std::string cast_name =
        AutoCast(input_info[i].name, input_info[i].dtype, max_index);
    casted_node.push_back(cast_name);
  }
  return casted_node;
}

}  // namespace paddle2onnx
