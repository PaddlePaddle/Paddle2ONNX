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
#include <fstream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "include/deploy/common/blob.h"
#include "include/deploy/common/config.h"
#include "include/deploy/postprocess/util/clipper.h"

namespace Deploy {

struct PaddleOcrResult {
  std::vector<std::vector<std::vector<int>>> boxes;
  float cls_score;
  float crnn_score;
  int label;
  std::vector<std::string> str_res;
};

class PaddleOcrPostProc {
 public:
  void Init(const ConfigParser &parser);

  bool Run(const std::vector<DataBlob> &outputs,
          const std::vector<ShapeInfo> &shape_infos,
          std::vector<PaddleOcrResult> *ocr_results);

  bool GetRotateCropImage(
          const cv::Mat &srcimage,
          const std::vector<std::vector<std::vector<int>>> &boxes,
          std::vector<cv::Mat> *imgs);

 private:
  bool DetPostProc(const std::vector<DataBlob> &outputs,
          const std::vector<ShapeInfo> &shape_infos,
          std::vector<PaddleOcrResult> *ocr_results);

  bool ClsPostProc(const std::vector<DataBlob> &outputs,
          std::vector<PaddleOcrResult> *ocr_results);

  bool CrnnPostProc(const std::vector<DataBlob> &outputs,
          std::vector<PaddleOcrResult> *ocr_results);

  bool ReadDict(const std::string &path);

  bool BoxesFromBitmap(
          const cv::Mat &pred,
          const cv::Mat &bitmap,
          const float &box_thresh,
          const float &det_db_unclip_ratio,
          PaddleOcrResult *ocr_result);

  bool FilterTagDetRes(const ShapeInfo &shape_info,
          PaddleOcrResult *ocr_result);

  bool GetContourArea(const std::vector<std::vector<float>> &box,
                      const float unclip_ratio, float *distance);

  cv::RotatedRect UnClip(std::vector<std::vector<float>> box,
                         const float &unclip_ratio);

  float BoxScoreFast(std::vector<std::vector<float>> box_array, cv::Mat pred);

  std::vector<std::vector<int>> OrderPointsClockwise(
          std::vector<std::vector<int>> pts);

  std::vector<std::vector<float>> GetMiniBoxes(
          cv::RotatedRect box, float *ssid);

  static bool XsortInt(std::vector<int> a, std::vector<int> b);

  static bool XsortFp32(std::vector<float> a, std::vector<float> b);

  std::vector<std::vector<float>> Mat2Vector(cv::Mat mat);

  inline int _max(int a, int b) { return a >= b ? a : b; }

  inline int _min(int a, int b) { return a >= b ? b : a; }

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
  std::vector<std::string> label_list_;
  double cls_thresh_ = 0.9;
  double det_db_thresh_ = 0.3;
  double det_db_box_thresh_ = 0.5;
  double det_db_unclip_ratio_ = 2.0;
};

}  // namespace Deploy
