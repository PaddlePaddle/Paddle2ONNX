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

#include "include/deploy/common/blob.h"
#include "include/deploy/common/config.h"

namespace Deploy {

struct PaddleOcrResult {
  std::vector<std::vector<std::vector<int>>> boxes;
};

class PaddleOcrPostProc {
 public:
  void Init(const ConfigParser &parser);
  bool Run(const std::vector<DataBlob> &outputs,
          const std::vector<ShapeInfo> &shape_infos,
          std::vector<PaddleOcrResult> *ocr_results);

 private:
  bool DetPostProc(const std::vector<DataBlob> &outputs,
          std::vector<PaddleOcrResult> *ocr_results);

  bool BoxesFromBitmap(
          const cv::Mat &pred,
          const cv::Mat &bitmap,
          const float &box_thresh,
          const float &det_db_unclip_ratio,
          PaddleOcrResult *ocr_result);

  bool FilterTagDetRes(const ShapeInfo &shape_info,
          PaddleOcrResult *ocr_result);

  std::vector<std::vector<int>> OrderPointsClockwise(
          std::vector<std::vector<int>> *pts);

  template <class T> inline T clamp(T x, T min, T max) {
    if (x > max)
      return max;
    if (x < min)
      return min;
    return x;
  }

  inline float clampf(float x, float min, float max) {
    if (x > max)
      return max;
    if (x < min)
      return min;
    return x;
  }

  std::string model_arch_;
  double det_db_thresh_ = 0.3;
  double det_db_unclip_ratio_ = 2.0;
};

}  // namespace Deploy
