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

#include <vector>
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
        bool Run(const std::vector<DataBlob> &outputs, const std::vector<ShapeInfo> &shape_traces, std::vector<PaddleDetResult> *det_results);
    private:
        std::string model_arch_;
        std::map<int, std::string> labels_;
};

}//namespaces