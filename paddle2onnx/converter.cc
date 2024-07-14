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

#include "paddle2onnx/converter.h"

#include <fstream>
#include <iostream>
#include <set>
#include <string>

#include "paddle2onnx/mapper/exporter.h"
#include "paddle2onnx/optimizer/convert_fp32_to_fp16.h"

namespace paddle2onnx
{
  PADDLE2ONNX_DECL bool Export(
      const char *model_filename,
      const char *params_filename,
      char **out,
      int *out_size,
      int32_t opset_version,
      bool auto_upgrade_opset,
      bool verbose,
      bool enable_onnx_checker,
      bool enable_experimental_op,
      bool enable_optimize,
      const char *deploy_backend,
      char **calibration_cache,
      int *calibration_size,
      const char *external_file,
      bool *save_external,
      bool export_fp16_model,
      char **disable_fp16_op_types,
      int disable_fp16_op_types_count)
  {
    auto parser = PaddleParser();
    P2OLogger(verbose) << "Start to parsing Paddle model..." << std::endl;
    if (!parser.Init(model_filename, params_filename))
    {
      P2OLogger(verbose) << "Paddle model parsing failed." << std::endl;
      return false;
    }
    paddle2onnx::ModelExporter me;

    // Add disabled fp16 op information
    std::vector<std::string> disable_op_types;
    if (disable_fp16_op_types != nullptr && disable_fp16_op_types_count > 0)
    {
      for (int i = 0; i < disable_fp16_op_types_count; ++i)
      {
        std::string disable_op_type(disable_fp16_op_types[i],
                                    strlen(disable_fp16_op_types[i]));
        disable_op_types.push_back(disable_op_type);
      }
    }
    std::string calibration_str;
    std::string result = me.Run(
        parser, opset_version, auto_upgrade_opset, verbose, enable_onnx_checker,
        enable_experimental_op, enable_optimize, deploy_backend, &calibration_str,
        external_file, save_external, export_fp16_model, disable_op_types);
    if (result.empty())
    {
      P2OLogger(verbose) << "The exported ONNX model is invalid!" << std::endl;
      return false;
    }
    if (parser.is_quantized_model && "tensorrt" == std::string(deploy_backend) &&
        calibration_str.empty())
    {
      P2OLogger(verbose) << "Can not generate calibration cache for TensorRT "
                            "deploy backend when export quantize model."
                         << std::endl;
      return false;
    }
    *out_size = result.size();
    *out = new char[*out_size]();
    memcpy(*out, result.data(), *out_size);
    if (calibration_str.size())
    {
      *calibration_size = calibration_str.size();
      *calibration_cache = new char[*calibration_size]();
      memcpy(*calibration_cache, calibration_str.data(), *calibration_size);
    }
    return true;
  }

  PADDLE2ONNX_DECL bool Export(
      const void *model_buffer,
      int64_t model_size,
      const void *params_buffer,
      int64_t params_size,
      char **out,
      int *out_size,
      int32_t opset_version,
      bool auto_upgrade_opset,
      bool verbose,
      bool enable_onnx_checker,
      bool enable_experimental_op,
      bool enable_optimize,
      const char *deploy_backend,
      char **calibration_cache,
      int *calibration_size,
      const char *external_file,
      bool *save_external,
      bool export_fp16_model,
      char **disable_fp16_op_types,
      int disable_fp16_op_types_count)
  {
    auto parser = PaddleParser();
    P2OLogger(verbose) << "Start to parsing Paddle model..." << std::endl;
    if (!parser.Init(model_buffer, model_size, params_buffer, params_size))
    {
      P2OLogger(verbose) << "Paddle model parsing failed." << std::endl;
      return false;
    }
    paddle2onnx::ModelExporter me;

    // Add disabled fp16 op information
    std::vector<std::string> disable_op_types;
    if (disable_fp16_op_types != nullptr && disable_fp16_op_types_count > 0)
    {
      for (int i = 0; i < disable_fp16_op_types_count; ++i)
      {
        std::string disable_op_type(disable_fp16_op_types[i],
                                    strlen(disable_fp16_op_types[i]));
        disable_op_types.push_back(disable_op_type);
      }
    }
    std::string calibration_str;
    std::string result = me.Run(
        parser, opset_version, auto_upgrade_opset, verbose, enable_onnx_checker,
        enable_experimental_op, enable_optimize, deploy_backend, &calibration_str,
        external_file, save_external, export_fp16_model, disable_op_types);
    if (result.empty())
    {
      P2OLogger(verbose) << "The exported ONNX model is invalid!" << std::endl;
      return false;
    }

    if (parser.is_quantized_model && "tensorrt" == std::string(deploy_backend) &&
        calibration_str.empty())
    {
      P2OLogger(verbose) << "Can not generate calibration cache for TensorRT "
                            "deploy backend when export quantize model."
                         << std::endl;
      return false;
    }
    *out_size = result.size();
    *out = new char[*out_size]();
    memcpy(*out, result.data(), *out_size);
    if (calibration_str.size())
    {
      *calibration_size = calibration_str.size();
      *calibration_cache = new char[*calibration_size]();
      memcpy(*calibration_cache, calibration_str.data(), *calibration_size);
    }
    return true;
  }

  PADDLE2ONNX_DECL bool ConvertFP32ToFP16(const char *onnx_model, int model_size,
                                          char **out_model, int *out_model_size)
  {
    std::string onnx_proto(onnx_model, onnx_model + model_size);
    ONNX_NAMESPACE::ModelProto model;
    model.ParseFromString(onnx_proto);

    P2OLogger(true) << "Convert FP32 ONNX model to FP16." << std::endl;
    ConvertFp32ToFp16 convert;
    convert.Convert(&model);
    // save external data file for big model
    std::string external_data_file;
    if (model.ByteSizeLong() > INT_MAX)
    {
      external_data_file = "external_data";
    }
    paddle2onnx::ModelExporter me;
    if (external_data_file.size())
    {
      me.SaveExternalData(model.mutable_graph(), external_data_file);
    }
    // check model
    me.ONNXChecker(model, true);

    std::string result;
    if (!model.SerializeToString(&result))
    {
      P2OLogger(true)
          << "Error happenedd while optimizing the exported ONNX model."
          << std::endl;
      return false;
    }

    *out_model_size = result.size();
    *out_model = new char[*out_model_size]();
    memcpy(*out_model, result.data(), *out_model_size);
    return true;
  }

  ModelTensorInfo::~ModelTensorInfo()
  {
    if (shape != nullptr)
    {
      delete[] shape;
      shape = nullptr;
      rank = 0;
    }
  }
} // namespace paddle2onnx
