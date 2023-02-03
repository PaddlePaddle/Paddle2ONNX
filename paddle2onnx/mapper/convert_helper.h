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

#include <cmath>
#include <fstream>
#include <iomanip>

#include <onnx/shape_inference/implementation.h>
#include "paddle2onnx/mapper/mapper.h"
#include "paddle2onnx/parser/parser.h"
namespace paddle2onnx {

struct proto_node {
 public:
  std::string node_type;  // model, graph, node, arribute
  ONNX_NAMESPACE::ModelProto* model;
  ONNX_NAMESPACE::GraphProto* graph;
  ONNX_NAMESPACE::NodeProto* node;
  ONNX_NAMESPACE::AttributeProto* attr;

  explicit proto_node(ONNX_NAMESPACE::ModelProto new_model) {
    node_type = "model";
    model = &new_model;
  }

  explicit proto_node(ONNX_NAMESPACE::ModelProto* new_model) {
    node_type = "model";
    model = new_model;
  }

  explicit proto_node(ONNX_NAMESPACE::GraphProto new_graph) {
    node_type = "graph";
    graph = &new_graph;
  }

  explicit proto_node(ONNX_NAMESPACE::GraphProto* new_graph) {
    node_type = "graph";
    graph = new_graph;
  }

  explicit proto_node(ONNX_NAMESPACE::NodeProto new_node) {
    node_type = "node";
    node = &new_node;
  }

  explicit proto_node(ONNX_NAMESPACE::NodeProto* new_node) {
    node_type = "node";
    node = new_node;
  }

  explicit proto_node(ONNX_NAMESPACE::AttributeProto new_attribute) {
    node_type = "attribute";
    attr = &new_attribute;
  }

  explicit proto_node(ONNX_NAMESPACE::AttributeProto* new_attribute) {
    node_type = "attribute";
    attr = new_attribute;
  }
};

struct ConvertFp32ToFp16 {
 public:
  ConvertFp32ToFp16(const float& min_positive_val = 1e-7,
                    const float& max_finite_val = 1e4,
                    const bool& keep_io_types = false,
                    const bool& disable_shape_infer = false,
                    const std::vector<std::string>& op_block_list = {},
                    const std::vector<std::string>& node_block_list = {}) {
    min_positive_val_ = min_positive_val;
    max_finite_val_ = max_finite_val;
    keep_io_types_ = keep_io_types;
    disable_shape_infer_ = disable_shape_infer;
    op_block_list_ = op_block_list;
    node_block_list_ = node_block_list;
  }

  void convert(ONNX_NAMESPACE::ModelProto& model);

  std::shared_ptr<ONNX_NAMESPACE::NodeProto> MakeCastNode(
      const std::string& op_name, const std::vector<std::string>& inputs,
      const std::vector<std::string>& outputs, int32_t to_dtype);

  std::shared_ptr<ONNX_NAMESPACE::ValueInfoProto> MakeValueInfoFromTensor(
      const ONNX_NAMESPACE::TensorProto& tensor);

  void KeepIoType(ONNX_NAMESPACE::ModelProto& model);

  void ConvertAttribute(ONNX_NAMESPACE::ModelProto& model);

  void ConvertTensorFloatToFloat16(ONNX_NAMESPACE::TensorProto* tensor);

  bool GetTensorValue(const ONNX_NAMESPACE::TensorProto& tensor,
                      std::vector<float>* value);

  void SortNodes(ONNX_NAMESPACE::ModelProto& model);

 private:
  float min_positive_val_ = 1e-7;
  float max_finite_val_ = 1e4;
  bool keep_io_types_ = false;
  bool disable_shape_infer_ = false;
  std::vector<std::string> op_block_list_ = {};
  std::vector<std::string> node_block_list_ = {};

  std::map<std::string, std::string> name_mapping;
  std::vector<std::string> graph_io_to_skip;
  std::vector<ONNX_NAMESPACE::ValueInfoProto*> value_info_list;
  std::vector<std::string> io_casts;

  std::vector<ONNX_NAMESPACE::NodeProto*> node_list;

  std::vector<proto_node> queue;
  std::vector<proto_node> next_level;

  int64_t name_index = 0;
  std::string GenName(const std::string& prefix);

  std::vector<std::string> DEFAULT_OP_BLOCK_LIST = {"ArrayFeatureExtractor",
                                                    "Binarizer",
                                                    "CastMap",
                                                    "CategoryMapper",
                                                    "DictVectorizer",
                                                    "FeatureVectorizer",
                                                    "Imputer",
                                                    "LabelEncoder",
                                                    "LinearClassifier",
                                                    "LinearRegressor",
                                                    "Normalizer",
                                                    "OneHotEncoder",
                                                    "RandomUniformLike",
                                                    "SVMClassifier",
                                                    "SVMRegressor",
                                                    "Scaler",
                                                    "TreeEnsembleClassifier",
                                                    "TreeEnsembleRegressor",
                                                    "ZipMap",
                                                    "NonMaxSuppression",
                                                    "TopK",
                                                    "RoiAlign",
                                                    "Resize",
                                                    "Range",
                                                    "CumSum",
                                                    "Min",
                                                    "Max",
                                                    "Upsample"};
};
}  // namespace paddle2onnx
