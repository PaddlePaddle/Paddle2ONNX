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
#include <algorithm>
#include <utility>

#include "include/deploy/common/blob.h"
#include "include/deploy/common/config.h"

namespace Deploy {

template <class T>
struct Mask {
  // raw data of mask
  std::vector<T> data;
  // the shape of mask
  std::vector<int> shape;
  void clear() {
    data.clear();
    shape.clear();
  }
};


struct Box {
  int category_id;
  // category label this box belongs to
  std::string category;
  // confidence score
  float score;
  std::vector<float> coordinate;
  Mask<int> mask;
};

struct PaddleDetResult {
  // target boxes
  std::vector<Box> boxes;
  int mask_resolution;
  void clear() {
      boxes.clear();
  }
};

class PaddleDetPostProc {
 public:
  void Init(const ConfigParser &parser);
  bool Run(const std::vector<DataBlob> &outputs,
          const std::vector<ShapeInfo> &shape_infos,
          const bool use_cpu_nms,
          std::vector<PaddleDetResult> *det_results);

 private:
  template <class T>
  bool SortScorePairDescend(const std::pair<float, T>& pair1,
                            const std::pair<float, T>& pair2) {
    return pair1.first > pair2.first;
  }

  bool DetPostNonNms(const std::vector<DataBlob> &outputs,
          const std::vector<ShapeInfo> &shape_infos,
          std::vector<PaddleDetResult> *det_results);

  bool DetPostWithNms(const std::vector<DataBlob> &outputs,
                      const std::vector<ShapeInfo> &shape_infos,
                      std::vector<PaddleDetResult> *det_results);

  void NMSFast(const DataBlob &score_blob,
              const DataBlob &box_blob,
              const int &i,
              const int &j,
              std::vector<int> *selected_indices);

  void GetMaxScoreIndex(const std::vector<float> &scores,
                        const float &threshold, const int &top_k,
                        std::vector<std::pair<float, int>> *sorted_indices);

  bool SortScorePairDescend(const std::pair<float, int>& pair1,
                          const std::pair<float, int>& pair2);

  float JaccardOverlap(const float* box1,
                      const float* box2,
                      const bool normalized);

  float BBoxArea(const float* box, const bool normalized);

  std::string model_arch_;
  std::map<int, std::string> labels_;
};

}  // namespace Deploy
