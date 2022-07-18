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

#include "paddle2onnx/optimizer/paddle2onnx_optimizer.h"
#include <onnx/shape_inference/implementation.h>
#include <fstream>
#include "onnxoptimizer/optimize.h"
#include "paddle2onnx/optimizer/eliminate_non_transpose.h"
#include "paddle2onnx/optimizer/fuse_constant_cast.h"
#include "paddle2onnx/optimizer/fuse_constant_reshape.h"
#include "paddle2onnx/optimizer/fuse_constant_unsqueeze.h"
#include "paddle2onnx/optimizer/fuse_paddle_conv_bias.h"
#include "paddle2onnx/optimizer/fuse_unsqueeze_conv2d_squeeze.h"
#include "paddle2onnx/optimizer/replace_add_to_identity.h"
#include "paddle2onnx/optimizer/replace_mul_to_identity.h"
#include "paddle2onnx/utils/utils.h"

namespace ONNX_NAMESPACE {
namespace optimization {

bool OptimizePaddle2ONNX(const std::string& model_path,
                         const std::string& optimized_model_path,
                         const OptimizerOption& option) {
  std::ifstream fin(model_path, std::ios::in | std::ios::binary);
  if (!fin.is_open()) {
    P2OLogger(true)
        << "Failed to read model file: " << model_path
        << ", please make sure your model file or file path is valid."
        << std::endl;
    return false;
  }
  std::string contents;
  fin.seekg(0, std::ios::end);
  contents.clear();
  contents.resize(fin.tellg());
  fin.seekg(0, std::ios::beg);
  fin.read(&(contents.at(0)), contents.size());
  fin.close();

  auto model_proto = std::make_shared<ONNX_NAMESPACE::ModelProto>();
  if (!model_proto->ParseFromString(contents)) {
    P2OLogger(true) << "Failed to optimize this model." << std::endl;
    return false;
  }
  ONNX_NAMESPACE::optimization::Optimizer::passes
      .registerPass<ONNX_NAMESPACE::optimization::FuseConstantReshape>();
  ONNX_NAMESPACE::optimization::Optimizer::passes
      .registerPass<ONNX_NAMESPACE::optimization::FuseConstantUnsqueeze>();
  ONNX_NAMESPACE::optimization::Optimizer::passes
      .registerPass<ONNX_NAMESPACE::optimization::FusePaddleConvBias>();
  ONNX_NAMESPACE::optimization::Optimizer::passes
      .registerPass<ONNX_NAMESPACE::optimization::FuseUnsqueezeConv2dSqueeze>();
  ONNX_NAMESPACE::optimization::Optimizer::passes
      .registerPass<ONNX_NAMESPACE::optimization::EliminateNonTranspose>();
  ONNX_NAMESPACE::optimization::Optimizer::passes
      .registerPass<ONNX_NAMESPACE::optimization::FuseConstantCast>();
  ONNX_NAMESPACE::optimization::Optimizer::passes
      .registerPass<ONNX_NAMESPACE::optimization::ReplaceMulToIdentity>();
  ONNX_NAMESPACE::optimization::Optimizer::passes
      .registerPass<ONNX_NAMESPACE::optimization::ReplaceAddToIdentity>();

  auto optimized_model_proto = ONNX_NAMESPACE::optimization::Optimize(
      *(model_proto.get()), option.passes);
  std::string optimized_model_str;
  if (!optimized_model_proto.SerializeToString(&optimized_model_str)) {
    P2OLogger(true) << "Failed to serialize the optimized model protobuf."
                    << std::endl;
    return false;
  }

  std::fstream out(optimized_model_path, std::ios::out | std::ios::binary);
  if (!out) {
    P2OLogger(true) << "Failed to write the optimized model to disk at "
                    << optimized_model_path << "." << std::endl;
    return false;
  }
  out << optimized_model_str;
  out.close();
  return true;
}

bool OptimizePaddle2ONNX(
    const std::string& model_path, const std::string& optimized_model_path,
    const std::map<std::string, std::vector<int>>& shape_infos,
    const OptimizerOption& option) {
  std::ifstream fin(model_path, std::ios::in | std::ios::binary);
  if (!fin.is_open()) {
    P2OLogger(true)
        << "Failed to read model file: " << model_path
        << ", please make sure your model file or file path is valid."
        << std::endl;
    return false;
  }
  std::string contents;
  fin.seekg(0, std::ios::end);
  contents.clear();
  contents.resize(fin.tellg());
  fin.seekg(0, std::ios::beg);
  fin.read(&(contents.at(0)), contents.size());
  fin.close();

  auto model_proto = std::make_shared<ONNX_NAMESPACE::ModelProto>();
  if (!model_proto->ParseFromString(contents)) {
    P2OLogger(true) << "Failed to optimize this model." << std::endl;
    return false;
  }

  if (shape_infos.size() > 0) {
    // reinfer shape for this onnx model
    auto graph = model_proto->mutable_graph();
    // clear all the type info of outputs
    auto output_size = graph->output_size();
    for (size_t i = 0; i < output_size; ++i) {
      graph->mutable_output(i)->clear_type();
    }
    // reset type info of inputs
    auto input_size = graph->input_size();
    for (size_t i = 0; i < input_size; ++i) {
      auto input_name = graph->input(i).name();
      auto iter = shape_infos.find(input_name);
      if (iter != shape_infos.end()) {
        auto tensor_type_proto =
            graph->mutable_input(i)->mutable_type()->mutable_tensor_type();
        tensor_type_proto->clear_shape();
        auto shape = tensor_type_proto->mutable_shape();
        for (auto& dim : iter->second) {
          shape->add_dim()->set_dim_value(dim);
        }
      }
    }

    try {
      shape_inference::InferShapes(*(model_proto.get()));
    } catch (const std::exception& e) {
      P2OLogger(true) << "[ERROR] Failed to reinfer shape for this model."
                      << std::endl;
      P2OLogger(true) << e.what() << std::endl;
      return false;
    }
  }

  ONNX_NAMESPACE::optimization::Optimizer::passes
      .registerPass<ONNX_NAMESPACE::optimization::FuseConstantReshape>();
  ONNX_NAMESPACE::optimization::Optimizer::passes
      .registerPass<ONNX_NAMESPACE::optimization::FuseConstantUnsqueeze>();
  ONNX_NAMESPACE::optimization::Optimizer::passes
      .registerPass<ONNX_NAMESPACE::optimization::FusePaddleConvBias>();
  ONNX_NAMESPACE::optimization::Optimizer::passes
      .registerPass<ONNX_NAMESPACE::optimization::FuseUnsqueezeConv2dSqueeze>();
  ONNX_NAMESPACE::optimization::Optimizer::passes
      .registerPass<ONNX_NAMESPACE::optimization::EliminateNonTranspose>();
  ONNX_NAMESPACE::optimization::Optimizer::passes
      .registerPass<ONNX_NAMESPACE::optimization::FuseConstantCast>();
  ONNX_NAMESPACE::optimization::Optimizer::passes
      .registerPass<ONNX_NAMESPACE::optimization::ReplaceMulToIdentity>();
  ONNX_NAMESPACE::optimization::Optimizer::passes
      .registerPass<ONNX_NAMESPACE::optimization::ReplaceAddToIdentity>();

  auto optimized_model_proto = ONNX_NAMESPACE::optimization::Optimize(
      *(model_proto.get()), option.passes);
  std::string optimized_model_str;
  if (!optimized_model_proto.SerializeToString(&optimized_model_str)) {
    P2OLogger(true) << "Failed to serialize the optimized model protobuf."
                    << std::endl;
    return false;
  }

  std::fstream out(optimized_model_path, std::ios::out | std::ios::binary);
  if (!out) {
    P2OLogger(true) << "Failed to write the optimized model to disk at "
                    << optimized_model_path << "." << std::endl;
    return false;
  }
  out << optimized_model_str;
  out.close();
  return true;
}
}
}  // namespace paddle2onnx
