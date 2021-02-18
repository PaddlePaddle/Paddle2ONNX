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

#pragma once

#include <string>
#include <iostream>

#include "yaml-cpp/yaml.h"



namespace Deploy {

class ConfigParser {
  public:
    ConfigParser() {}

    ~ConfigParser() {}

    bool Load(const std::string &cfg_file, const std::string &pp_type);

    template <typename T>
    const T& Get(const std::string &name) const {
      return config_[name].as<T>();
    }

    YAML::Node GetNode(const std::string &node_name) const;
    
  private:
    bool DetParser(const YAML::Node &det_config);

    bool DetParserTransforms(const YAML::Node &preprocess_op);

    YAML::Node config_;
};

}//namespace
