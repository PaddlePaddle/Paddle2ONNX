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


#include "include/deploy/preprocess/ppclas_pre_proc.h"

namespace Deploy {

bool PaddleClasPreProc::Init(const ConfigParser &parser) {
  BuildTransform(parser);
}

bool PaddleClasPreProc::Run(const std::vector<cv::Mat> &imgs,
                          std::vector<DataBlob> *inputs) {
  inputs->clear();
  shape_infos->clear();
  int batchsize = imgs.size();
  DataBlob img_blob;
  std::vector<cv::Mat> images;
  images.assign(imgs.begin(), imgs.end());
  if (!RunTransform(&images)) {
    std::cerr << "Apply transforms to image failed!" << std::endl;
    return false;
  }
  int img_w = images[0].cols;
  int img_h = images[0].rows;
  int input_size = img_w * img_h * 3;
  img_blob.data.resize(input_size * batchsize * sizeof(float));
  for (int i=0; i < batchsize; i++) {
    // img data for input
    memcpy(img_blob.data.data() + i * input_size * sizeof(float),
            images[i].data, input_size * sizeof(float));
  }
  img_blob.shape = {batchsize, 3, img_h, img_w};
  img_blob.dtype = 0;
  img_blob.name = "inputs";
  inputs->push_back(std::move(img_blob));
}

}  // namespace Deploy

