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

#include "include/deploy/preprocess/ppseg_pre_proc.h"


namespace Deploy {

bool PaddleSegPreProc::Init(const ConfigParser &parser) {
  BuildTransform(parser);
}

bool PaddleSegPreProc::Run(const std::vector<cv::Mat> &imgs,
                          std::vector<DataBlob> *inputs,
                          std::vector<ShapeInfo> *shape_infos) {
  inputs->clear();
  shape_infos->clear();
  int batchsize = imgs.size();
  DataBlob img_blob;
  ShapeInfer(imgs, shape_infos);
  std::vector<int> max_shape = GetMaxSize();
  std::vector<cv::Mat> images;
  images.assign(imgs.begin(), imgs.end());
  if (!RunTransform(&images)) {
    std::cerr << "Apply transforms to image failed!" << std::endl;
    return false;
  }
  int input_size = max_shape[0] * max_shape[1] * 3;
  img_blob.data.resize(input_size * batchsize * sizeof(float));
  for (int i=0; i < batchsize; i++) {
    // img data for input
    memcpy(img_blob.data.data() + i * input_size * sizeof(float),
            images[i].data, input_size * sizeof(float));
  }
  img_blob.shape = {batchsize, 3, max_shape[1], max_shape[0]};
  img_blob.dtype = 0;
  img_blob.name = "generated_tensor_0";
  inputs->push_back(std::move(img_blob));
}

}  // namespace Deploy
