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

#include "include/deploy/preprocess/preprocess.h"

namespace Deploy {

bool BasePreprocess::BuildTransform(const ConfigParser &parser) {
  transforms.clear();
  std::string transform_node = "transforms";
  YAML::Node transforms_node = parser.GetNode(transform_node);
  for (YAML::const_iterator it = transforms_node.begin();
      it != transforms_node.end(); ++it) {
    std::string name = it->first.as<std::string>();
    std::shared_ptr<Transform> transform = CreateTransform(name);
    transform->Init(it->second);
    transforms.push_back(transform);
  }
}

bool BasePreprocess::RunTransform(std::vector<cv::Mat> *imgs) {
  for (int i=0; i < transforms.size(); i++) {
    if (!transforms[i]->Run(imgs)) {
      std::cerr << "Run transforms to image failed!" << std::endl;
      return false;
    }
  }
  Padding batch_padding;
  batch_padding.Run(imgs, max_w_, max_h_);
  return true;
}

bool BasePreprocess::ShapeInfer(const std::vector<cv::Mat> &imgs,
                                std::vector<ShapeInfo> *shape_traces) {
  int batchsize = imgs.size();
  for (int i = 0; i < batchsize; i++) {
    ShapeInfo im_shape;
    std::vector<int> origin_size = {imgs[i].cols, imgs[i].rows};
    im_shape.transform_order.push_back("Origin");
    im_shape.shape.push_back(origin_size);
    if (!transforms[i]->ShapeInfer(&im_shape)) {
      std::cerr << "Apply shape inference failed!" << std::endl;
      return false;
    }
    std::vector<int> final_shape = im_shape.shape.back();
    if (final_shape[0] > max_w_) {
      max_w_ = final_shape[0];
    }
    if (final_shape[1] > max_h_) {
      max_h_ = final_shape[1];
    }
    shape_traces->push_back(std::move(im_shape));
  }
  for (int i = 0; i < batchsize; i++) {
    std::vector<int> max_shape = GetMaxSize();
    (*shape_traces)[i].shape.push_back(std::move(max_shape));
    (*shape_traces)[i].transform_order.push_back("Padding");
  }
  return true;
}

std::vector<int> BasePreprocess::GetMaxSize() {
  std::vector<int> max_shape = {max_w_, max_h_};
  return max_shape;
}

std::shared_ptr<Transform> BasePreprocess::CreateTransform(
    const std::string& transform_name) {
  if (transform_name == "Normalize") {
    return std::make_shared<Normalize>();
  } else if (transform_name == "ResizeByShort") {
    return std::make_shared<ResizeByShort>();
  } else if (transform_name == "ResizeByLong") {
    return std::make_shared<ResizeByLong>();
  } else if (transform_name == "CenterCrop") {
    return std::make_shared<CenterCrop>();
  } else if (transform_name == "Permute") {
    return std::make_shared<Permute>();
  } else if (transform_name == "Resize") {
    return std::make_shared<Resize>();
  } else if (transform_name == "Padding") {
    return std::make_shared<Padding>();
  } else if (transform_name == "Clip") {
    return std::make_shared<Clip>();
  } else if (transform_name == "RGB2BGR") {
    return std::make_shared<RGB2BGR>();
  } else if (transform_name == "BGR2RGB") {
    return std::make_shared<BGR2RGB>();
  } else if (transform_name == "Convert") {
    return std::make_shared<Convert>();
  } else {
    std::cerr << "There's unexpected transform(name='" << transform_name
              << "')." << std::endl;
    exit(-1);
  }
}

}  // namespace Deploy
