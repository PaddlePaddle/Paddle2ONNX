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

#include <vector>
#include <string>
#include <iostream>

#include "include/deploy/common/blob.h"
#include "include/deploy/engine/engine_config.h"

#include "paddle_inference_api.h"

#ifdef _WIN32
#define OS_PATH_SEP "\\"
#else
#define OS_PATH_SEP "/"
#endif

namespace Deploy {

class PaddleInferenceEngine{
  public:
    void Init(std::string model_dir, PaddleInferenceConfig &config);

    void Infer(std::vector<DataBlob> &inputs, std::vector<DataBlob> *outputs);
  private:
    std::unique_ptr<paddle::PaddlePredictor> predictor_;


};

}
