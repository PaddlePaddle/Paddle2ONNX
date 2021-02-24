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


#include "include/deploy/preprocess/ppdet_pre_proc.h"


namespace Deploy {

bool PaddleDetPreProc::Init(const ConfigParser &parser) {
  BuildTransform(parser);
  model_arch_ = parser.Get<std::string>("model_name");
}

bool PaddleDetPreProc::Run(const std::vector<cv::Mat> &imgs,
                          std::vector<DataBlob> *inputs,
                          std::vector<ShapeInfo> *shape_traces) {
  inputs->clear();
  shape_traces->clear();
  int batchsize = imgs.size();
  DataBlob img_blob;
  DataBlob im_size_blob;
  DataBlob im_info_blob;
  DataBlob im_shape_blob;
  DataBlob scale_factor_blob;
  ShapeInfer(imgs, shape_traces);
  std::vector<int> max_shape = GetMaxSize();
  std::vector<cv::Mat> images;
  images.assign(imgs.begin(), imgs.end());
  for (int i = 0; i < images.size(); i++) {
    cv::cvtColor(images[i], images[i], cv::COLOR_BGR2RGB);
  }
  if (!RunTransform(&images)) {
    std::cerr << "Apply transforms to image failed!" << std::endl;
    return false;
  }
  int input_size = max_shape[0] * max_shape[1] * 3;
  img_blob.data.resize(input_size * batchsize * sizeof(float));
  im_size_blob.data.resize(2 * batchsize * sizeof(int));
  im_info_blob.data.resize(3 * batchsize * sizeof(float));
  im_shape_blob.data.resize(3 * batchsize * sizeof(float));
  scale_factor_blob.data.resize(4 * batchsize * sizeof(float));
  for (int i=0; i < batchsize; i++) {
    // img data for input
    memcpy(img_blob.data.data() + i * input_size * sizeof(float),
            images[i].data, input_size * sizeof(float));
    // Additional information for input
    if (model_arch_ == "YOLO") {
      std::vector<int> origin_size =
            {(*shape_traces)[i].shape[0][1], (*shape_traces)[i].shape[0][0]};
      memcpy(im_size_blob.data.data() + i * 2 * sizeof(int),
            origin_size.data(), 2 * sizeof(int));
    }
    if (model_arch_ == "RCNN") {
      std::vector<float> im_info = (*shape_traces)[i].GetImInfo();
      std::vector<float> im_shape =
            {static_cast<float>((*shape_traces)[i].shape[0][1]),
            static_cast<float>((*shape_traces)[i].shape[0][0]),
            1};
      memcpy(im_info_blob.data.data() + i * 3 * sizeof(float),
            im_info.data(), 3 * sizeof(float));
      memcpy(im_shape_blob.data.data() + i * 3 * sizeof(float),
            im_shape.data(), 3 * sizeof(float));
    }
    if (model_arch_ == "RetinaNet" ||
        model_arch_ == "EfficientDet" ||
        model_arch_ == "FCOS") {
      std::vector<float> im_info = (*shape_traces)[i].GetImInfo();
      memcpy(im_info_blob.data.data() + i * 3 * sizeof(float),
            im_info.data(), 3 * sizeof(float));
    }
    if (model_arch_ == "TTFNet") {
      std::vector<float> scale = (*shape_traces)[i].GetScale();
      std::vector<float> scale_factor =
            {scale[0], scale[1], scale[0], scale[1]};
      memcpy(scale_factor_blob.data.data() + i * 4 * sizeof(float),
            scale_factor.data(), 4 * sizeof(float));
    }
  }
  // Feed img data for input
  img_blob.shape = {batchsize, 3, max_shape[1], max_shape[0]};
  img_blob.dtype = 0;
  img_blob.name = "image";
  inputs->push_back(std::move(img_blob));
  // Feed additional information for input
  if (model_arch_ == "YOLO") {
    im_size_blob.name = "im_size";
    im_size_blob.shape = {batchsize, 2};
    im_size_blob.dtype = 2;
    inputs->push_back(std::move(im_size_blob));
  }
  if (model_arch_ == "RCNN") {
    im_info_blob.name = "im_info";
    im_info_blob.shape = {batchsize, 3};
    im_info_blob.dtype = 0;
    inputs->push_back(std::move(im_info_blob));
    im_shape_blob.name = "im_shape";
    im_shape_blob.shape = {batchsize, 3};
    im_shape_blob.dtype = 0;
    inputs->push_back(std::move(im_shape_blob));
  }
  if (model_arch_ == "RetinaNet" ||
      model_arch_ == "EfficientDet" ||
      model_arch_ == "FCOS") {
    im_info_blob.name = "im_info";
    im_info_blob.shape = {batchsize, 3};
    im_info_blob.dtype = 0;
    inputs->push_back(std::move(im_info_blob));
  }
  if (model_arch_ == "TTFNet") {
    scale_factor_blob.name = "scale_factor";
    im_info_blob.shape = {batchsize, 4};
    im_info_blob.dtype = 0;
    inputs->push_back(std::move(scale_factor_blob));
  }
}

}  //  namespace Deploy
