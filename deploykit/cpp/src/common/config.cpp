// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include <string>
#include <iostream>

#include "yaml-cpp/yaml.h"

#include "include/deploy/common/config.h"

namespace Deploy {

bool ConfigParser::Load(const std::string &cfg_file,
                        const std::string &toolkit) {
  // Load config as a YAML::Node
  YAML::Node config = YAML::LoadFile(cfg_file);
  // Parser yaml file
  if (toolkit == "det") {
    if (!DetParser(config)) {
      std::cerr << "Fail to parser PaddleDection yaml file" << std::endl;
      return false;
    }
  } else if (toolkit == "paddle" || toolkit == "clas") {
    if (!CommonParser(config)) {
      std::cerr << "Fail to parser Paddle yaml file" << std::endl;
      return false;
    }
  } else if (toolkit == "ocr") {
    if (!OcrParser(config)) {
      std::cerr << "Fail to parser PaddleOCR yaml file" << std::endl;
      return false;
    }
  } else if (toolkit == "seg") {
    if (!SegParser(config)) {
      std::cerr << "Fail to parser PaddleSeg yaml file" << std::endl;
    }
  }
  return true;
}

YAML::Node ConfigParser::GetNode(const std::string &node_name) const {
  return config_[node_name];
}

bool ConfigParser::SegParser(const YAML::Node &seg_config) {
  config_["toolkit"] = "PaddleSeg";
  config_["toolkit_version"] = "Unknown";
  config_["transforms"]["RGB2BGR"]["is_rgb2bgr"] = true;
  YAML::Node preprocess_op = seg_config["Deploy"]["transforms"];
  for (const auto& item : preprocess_op) {
    std::string name = item.begin()->second.as<std::string>();
    if (name == "Normalize") {
      config_["transforms"]["Convert"]["dtype"] = "float";
      config_["transforms"]["Normalize"]["is_scale"] = true;
      for (int i = 0; i < 3; i++) {
        config_["transforms"]["Normalize"]["mean"].push_back(0.5);
        config_["transforms"]["Normalize"]["std"].push_back(0.5);
        config_["transforms"]["Normalize"]["min_val"].push_back(0);
        config_["transforms"]["Normalize"]["max_val"].push_back(255);
      }
    } else if (name == "Padding") {
      std::vector<int> target_size = item["target_size"].as<std::vector<int>>();
      config_["transforms"]["Padding"]["width"] = target_size[0];
      config_["transforms"]["Padding"]["height"] = target_size[1];
      for (int i = 0; i < 3; i++) {
        config_["transforms"]["Padding"]["im_padding_value"].push_back(127.5);
      }
    } else {
      std::cout << "can't parser: " << name << std::endl;
      return false;
    }
  }
  config_["transforms"]["Permute"]["is_permute"] = true;
  return true;
}

bool ConfigParser::OcrParser(const YAML::Node &ocr_config) {
  if (ocr_config["fix_shape"].as<bool>()) {
    config_["model_name"] = ocr_config["arch"].as<std::string>();
    config_["path"] = ocr_config["path"].as<std::string>();
    config_["toolkit"] = "PaddleOCR";
    config_["toolkit_version"] = "Unknown";
    if (ocr_config["arch"].as<std::string>() == "DET") {
      config_["det_db_thresh"] = ocr_config["det_db_thresh"].as<double>();
      config_["det_db_box_thresh"] =
        ocr_config["det_db_box_thresh"].as<double>();
      config_["det_db_unclip_ratio"] =
        ocr_config["det_db_unclip_ratio"].as<double>();
    }
    if (ocr_config["arch"].as<std::string>() == "CLS") {
      config_["cls_thresh"] = ocr_config["cls_thresh"].as<double>();
    }
    YAML::Node preprocess_op = ocr_config["transforms"];
    if (!OcrParserTransforms(preprocess_op)) {
      std::cerr << "Fail to parser PaddleOCR transforms failed" << std::endl;
      return false;
    }
  } else {
    config_ = ocr_config;
  }
  return true;
}

bool ConfigParser::OcrParserTransforms(const YAML::Node &preprocess_op) {
  for (const auto& item : preprocess_op) {
    std::string name = item.begin()->first.as<std::string>();
    if (name == "ResizeByLong") {
      if (config_["model_name"].as<std::string>() == "DET") {
        config_["transforms"]["Resize"]["width"] = 640;
        config_["transforms"]["Resize"]["height"] = 640;
      }
    } else if (name == "OcrResize") {
      if (config_["model_name"].as<std::string>() == "CLS") {
        config_["transforms"]["Resize"]["width"] = 100;
        config_["transforms"]["Resize"]["height"] = 32;
      }
      if (config_["model_name"].as<std::string>() == "CRNN") {
        config_["transforms"]["OcrTrtResize"]["width"] = 100;
        config_["transforms"]["OcrTrtResize"]["height"] = 32;
      }
    } else {
      config_["transforms"][name] = item.begin()->second;
    }
  }
}

bool ConfigParser::CommonParser(const YAML::Node &paddle_config) {
  if (!paddle_config["transforms"].IsDefined()) {
    std::cerr << "Fail to find transforms in Paddle yaml file" << std::endl;
    return false;
  }
  config_ = paddle_config;
  return true;
}

bool ConfigParser::DetParser(const YAML::Node &det_config) {
  config_["model_format"] = "Paddle";
  // arch support value:YOLO, SSD, RetinaNet, RCNN, Face
  if (det_config["arch"].IsDefined()) {
    config_["model_name"] = det_config["arch"].as<std::string>();
  } else {
    std::cerr << "Fail to find arch in PaddleDection yaml file" << std::endl;
    return false;
  }
  config_["toolkit"] = "PaddleDetection";
  config_["toolkit_version"] = "Unknown";

  if (det_config["label_list"].IsDefined()) {
    int i = 0;
    for (const auto& label : det_config["label_list"]) {
        config_["labels"][i] = label.as<std::string>();
        i++;
    }
  } else {
    std::cerr << "Fail to find label_list in  PaddleDection yaml file"
              << std::endl;
    return false;
  }
  // Preprocess support Normalize, Permute, Resize, PadStride, Convert
  if (det_config["Preprocess"].IsDefined()) {
    YAML::Node preprocess_info = det_config["Preprocess"];
    for (const auto& preprocess_op : preprocess_info) {
        if (!DetParserTransforms(preprocess_op)) {
            std::cerr << "Fail to parser PaddleDetection transforms"
                      << std::endl;
            return false;
        }
    }
  } else {
    std::cerr << "Fail to find Preprocess in  PaddleDection yaml file"
              << std::endl;
    return false;
  }
  return true;
}

bool ConfigParser::DetParserTransforms(const YAML::Node &preprocess_op) {
  if (preprocess_op["type"].as<std::string>() == "Normalize") {
    config_["transforms"]["Convert"]["dtype"] = "float";
    std::vector<float> mean = preprocess_op["mean"].as<std::vector<float>>();
    std::vector<float> std = preprocess_op["std"].as<std::vector<float>>();
    config_["transforms"]["Normalize"]["is_scale"] =
                preprocess_op["is_scale"].as<bool>();
    for (int i = 0; i < mean.size(); i++) {
      config_["transforms"]["Normalize"]["mean"].push_back(mean[i]);
      config_["transforms"]["Normalize"]["std"].push_back(std[i]);
      config_["transforms"]["Normalize"]["min_val"].push_back(0);
      config_["transforms"]["Normalize"]["max_val"].push_back(255);
    }
    return true;
  } else if (preprocess_op["type"].as<std::string>() == "Permute") {
    config_["transforms"]["Permute"]["is_permute"] = true;
    if (preprocess_op["to_bgr"].as<bool>() == true) {
      config_["transforms"]["RGB2BGR"]["is_rgb2bgr"] = true;
    }
    return true;
  } else if (preprocess_op["type"].as<std::string>() == "Resize") {
    int max_size = preprocess_op["max_size"].as<int>();
    if (max_size !=0 && (config_["model_name"].as<std::string>() == "RCNN"
        || config_["model_name"].as<std::string>() == "RetinaNet")) {
      config_["transforms"]["ResizeByShort"]["target_size"] =
                  preprocess_op["target_size"].as<int>();
      config_["transforms"]["ResizeByShort"]["max_size"] = max_size;
      config_["transforms"]["ResizeByShort"]["interp"] =
                  preprocess_op["interp"].as<int>();
      if (preprocess_op["image_shape"].IsDefined()) {
        config_["transforms"]["Padding"]["width"] = max_size;
        config_["transforms"]["Padding"]["height"] = max_size;
      }
    } else {
      config_["transforms"]["Resize"]["width"] =
                  preprocess_op["target_size"].as<int>();
      config_["transforms"]["Resize"]["height"] =
                  preprocess_op["target_size"].as<int>();
      config_["transforms"]["Resize"]["interp"] =
                  preprocess_op["interp"].as<int>();
      config_["transforms"]["Resize"]["max_size"] = max_size;
    }
    return true;
  } else if (preprocess_op["type"].as<std::string>() == "PadStride") {
    config_["transforms"]["Padding"]["stride"] =
                preprocess_op["stride"].as<int>();
    return true;
  } else {
    std::cerr << preprocess_op["type"].as<std::string>()
              << " :Can't parser" << std::endl;
    return false;
  }
}

}  // namespace Deploy
