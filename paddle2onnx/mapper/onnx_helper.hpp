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

#pragma once

#include <onnx/onnx_pb.h>
#include <vector>
#include "paddle2onnx/mapper/register_mapper.hpp"
#include "paddle2onnx/parser/parser.hpp"

namespace paddle2onnx {

class OnnxHelper {
 public:
  std::vector<std::shared_ptr<ONNX_NAMESPACE::NodeProto>> nodes;

  void Clear() { nodes.clear(); }

  std::shared_ptr<ONNX_NAMESPACE::NodeProto> MakeNode(
      const std::string& op_type, const std::vector<std::string>& inputs,
      const std::vector<std::string>& outputs);
  // we use this function to generate some temporary node
  // we do not need to define the outputs, because the outputs
  // is generate by MapperHelper, which will make sure there's no
  // name confict problem
  // the parameter `num_outputs` will define the number of output names
  std::shared_ptr<ONNX_NAMESPACE::NodeProto> MakeNode(
      const std::string& op_type, const std::vector<std::string>& inputs,
      int num_outputs = 1);

  template <typename T>
  std::shared_ptr<ONNX_NAMESPACE::NodeProto> MakeConstant(
      const std::vector<int64_t>& shape,
      ONNX_NAMESPACE::TensorProto_DataType dtype, T value);

  template <typename T>
  std::shared_ptr<ONNX_NAMESPACE::NodeProto> MakeConstant(
      const std::string& name, const std::vector<int64_t>& shape,
      ONNX_NAMESPACE::TensorProto_DataType dtype, T value);
};

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
  attr->set_i(int(dtype));
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

template <typename T>
std::shared_ptr<ONNX_NAMESPACE::NodeProto> OnnxHelper::MakeConstant(
    const std::vector<int64_t>& shape,
    ONNX_NAMESPACE::TensorProto_DataType dtype, T value) {
  auto node = std::make_shared<ONNX_NAMESPACE::NodeProto>();
  node->set_op_type("Constant");
  auto name = MapperHelper::Get()->GenName("const");
  node->add_output(name);
  auto attr = node->add_attribute();
  attr->set_name("value");
  attr->set_type(ONNX_NAMESPACE::AttributeProto::TENSOR);
  auto tensor = attr->mutable_t();
  tensor->set_name(name);

  int numel = 1;
  for (size_t i = 0; i < shape.size(); ++i) {
    tensor->add_dims(shape[i]);
    numel *= shape[i];
  }
  tensor->set_data_type(dtype);
  if (dtype == ONNX_NAMESPACE::TensorProto::FLOAT) {
    std::vector<float> data(numel, static_cast<float>(value));
    tensor->set_raw_data(std::string((const char*)(data.data()), numel * 4));
  } else if (dtype == ONNX_NAMESPACE::TensorProto::DOUBLE) {
    std::vector<double> data(numel, static_cast<double>(value));
    tensor->set_raw_data(std::string((const char*)(data.data()), numel * 8));
  } else if (dtype == ONNX_NAMESPACE::TensorProto::INT64) {
    std::vector<int64_t> data(numel, static_cast<int64_t>(value));
    tensor->set_raw_data(std::string((const char*)(data.data()), numel * 8));
  } else {
    Assert(false,
           "Only support data type of FLOAT/DOUBLE/INT64 in MakeConstant "
           "function.");
  }
  return node;
}

template <typename T>
std::shared_ptr<ONNX_NAMESPACE::NodeProto> OnnxHelper::MakeConstant(
    const std::string& name, const std::vector<int64_t>& shape,
    ONNX_NAMESPACE::TensorProto_DataType dtype, T value) {
  auto node = std::make_shared<ONNX_NAMESPACE::NodeProto>();
  node->set_op_type("Constant");
  node->add_output(name);
  auto attr = node->add_attribute();
  attr->set_name("value");
  attr->set_type(ONNX_NAMESPACE::AttributeProto::TENSOR);
  auto tensor = attr->mutable_t();
  tensor->set_name(name);

  int numel = 1;
  for (size_t i = 0; i < shape.size(); ++i) {
    tensor->add_dims(shape[i]);
    numel *= shape[i];
  }
  tensor->set_data_type(dtype);
  if (dtype == ONNX_NAMESPACE::TensorProto::FLOAT) {
    std::vector<float> data(numel, static_cast<float>(value));
    tensor->set_raw_data(std::string((const char*)(data.data()), numel * 4));
  } else if (dtype == ONNX_NAMESPACE::TensorProto::DOUBLE) {
    std::vector<double> data(numel, static_cast<double>(value));
    tensor->set_raw_data(std::string((const char*)(data.data()), numel * 8));
  } else if (dtype == ONNX_NAMESPACE::TensorProto::INT64) {
    std::vector<int64_t> data(numel, static_cast<int64_t>(value));
    tensor->set_raw_data(std::string((const char*)(data.data()), numel * 8));
  } else {
    Assert(false,
           "Only support data type of FLOAT/DOUBLE/INT64 in MakeConstant "
           "function.");
  }
  return node;
}

}  // namespace paddle2onnx
