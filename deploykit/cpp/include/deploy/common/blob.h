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
#include <iostream>

namespace Deploy {

class DataBlob{
 public:
  // data
  std::vector<char> data;

  // data name
  std::string name;

  // data shape
  std::vector<int> shape;

    /*
    data dtype
    0: FLOAT32
    1: INT64
    2: INT32
    3: UINT8
    */
  int dtype;

  // Lod Info
  std::vector<std::vector<size_t>> lod;
};

class ShapeInfo{
 public:
  // shape trace
  std::vector<std::vector<int> > shape;

  // transform order
  std::vector<std::string> transform_order;

  // transform index
  int GetIndex(const std::string &name) {
    std::vector<std::string>::iterator it =
      std::find(transform_order.begin(), transform_order.end(), name);
    if (it != transform_order.end()) {
      return (it - transform_order.begin());
    } else {
      std::cerr << "find " << name << " failed" << std::endl;
      return -1;
    }
  }
  // im_info = {h, w, scale_w}
  std::vector<float> GetImInfo() {
    int transforms_num = shape.size() - 1;
    float new_w = static_cast<float>(shape[transforms_num][0]);
    float origin_w = static_cast<float>(shape[0][0]);
    float scale_w = new_w / origin_w;
    std::vector<float> im_info =
          {static_cast<float>(shape[transforms_num][1]),
          static_cast<float>(shape[transforms_num][0]),
          scale_w};
    return im_info;
  }

  // scale = {scale_w, scale_h}
  std::vector<float> GetScale() {
    int transforms_num = shape.size() - 1;
    float new_w = static_cast<float>(shape[transforms_num][0]);
    float new_h = static_cast<float>(shape[transforms_num][1]);
    float origin_w = static_cast<float>(shape[0][0]);
    float origin_h = static_cast<float>(shape[0][1]);
    float scale_w = new_w / origin_w;
    float scale_h = new_h / origin_h;
    std::vector<float> scale = {scale_w, scale_h};
    return scale;
  }
};

}  // namespace Deploy
