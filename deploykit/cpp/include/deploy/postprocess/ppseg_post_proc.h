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
#include <map>
#include <string>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "include/deploy/common/blob.h"
#include "include/deploy/common/config.h"

namespace Deploy {

template <class T>
struct Mask {
  // raw data of mask
  std::vector<T> data;
  // the shape of mask
  std::vector<int> shape;
};

struct PaddleSegResult{
  // represent label of each pixel on image matrix
  Mask<int64_t> label_map;
  // represent score of each pixel on image matrix
  Mask<float> score_map;
};

class PaddleSegPostProc {
 public:
  void Init(const ConfigParser &parser) {}

  bool Run(const std::vector<DataBlob> &outputs,
          const std::vector<ShapeInfo> &shape_infos,
          std::vector<PaddleSegResult> *seg_results);
};

}  // namespace Deploy
