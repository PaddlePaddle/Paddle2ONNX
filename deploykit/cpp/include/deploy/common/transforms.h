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
#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "blob.h"
#include "yaml-cpp/yaml.h"

namespace Deploy {

class Transform {
  public:

    virtual void Init(const YAML::Node& item) = 0;
  
    virtual bool ShapeInfer(ShapeInfo* shape) = 0;
  
    virtual bool Run(std::vector<cv::Mat> *ims) = 0;
};

class Normalize : public Transform {
  public:
    virtual void Init(const YAML::Node& item) {
      mean_ = item["mean"].as<std::vector<float>>();
      std_ = item["std"].as<std::vector<float>>();
      if (item["is_scale"].IsDefined()) {
        is_scale_ = item["is_scale"];
      }
      if (item["min_val"].IsDefined()) {
        min_val_ = item["min_val"].as<std::vector<float>>();
      } else {
        min_val_ = std::vector<float>(mean_.size(), 0.);
      }
      if (item["max_val"].IsDefined()) {
        max_val_ = item["max_val"].as<std::vector<float>>();
      } else {
        max_val_ = std::vector<float>(mean_.size(), 255.);
      }
    }
    virtual bool Run(std::vector<cv::Mat> *ims);
    virtual bool ShapeInfer(ShapeInfo* shape);
  
  private:
    bool is_scale_;
    std::vector<float> mean_;
    std::vector<float> std_;
    std::vector<float> min_val_;
    std::vector<float> max_val_;
};

/*interp_: std::vector<int> interpolations = {
  cv::INTER_LINEAR, 
  cv::INTER_NEAREST, 
  cv::INTER_AREA, 
  cv::INTER_CUBIC, 
  cv::INTER_LANCZOS4}*/
class ResizeByShort : public Transform {
  public:
    virtual void Init(const YAML::Node& item) {
      target_size_ = item["target_size"].as<int>();
      if (item["interp"].IsDefined()) {
        interp_ = item["interp"].as<int>();
      }
      else {
        interp_ = 0;
      }
      if (item["max_size"].IsDefined()) {
        max_size_ = item["max_size"].as<int>();
      } else {
        max_size_ = -1;
      }
    }
  virtual bool Run(std::vector<cv::Mat> *ims);
  virtual bool ShapeInfer(ShapeInfo* shape);

 private:
    float GenerateScale(const int origin_w, const int origin_h);
    int target_size_;
    int max_size_;
    int interp_;
};


class ResizeByLong : public Transform {
  public:
  virtual void Init(const YAML::Node& item) {
    target_size_ = item["target_size"].as<int>();
    if (item["interp"].IsDefined()) {
        interp_ = item["interp"].as<int>();
    } else {
      interp_ = 0;
    }
    if (item["max_size"].IsDefined()) {
        max_size_ = item["max_size"].as<int>();
    } else {
      max_size_ = -1;
    }
  }
  virtual bool Run(std::vector<cv::Mat> *ims);
  virtual bool ShapeInfer(ShapeInfo* shape);

  private:
  float GenerateScale(const int origin_w, const int origin_h);
  int target_size_;
  int max_size_;
  int interp_;
};

class Resize : public Transform {
 public:
  virtual void Init(const YAML::Node& item) {
    if (item["interp"].IsDefined()) {
      interp_ = item["interp"].as<int>();
    }
    height_ = item["height"].as<int>();
    width_ = item["width"].as<int>();
    if (height_ <= 0 || width_ <= 0) {
      std::cerr << "[Resize] target_size should greater than 0" << std::endl;
      exit(-1);
    }
  }
  virtual bool Run(std::vector<cv::Mat> *ims);
  virtual bool ShapeInfer(ShapeInfo* shape);

 private:
  int height_;
  int width_;
  int interp_;
};

class BGR2RGB : public Transform {
  public:
    virtual void Init(const YAML::Node& item) {
    }
    virtual bool Run(std::vector<cv::Mat> *ims);
    virtual bool ShapeInfer(ShapeInfo* shape);
};

class RGB2BGR : public Transform {
  public:
    virtual void Init(const YAML::Node& item) {
    }
    virtual bool Run(std::vector<cv::Mat> *ims);
    virtual bool ShapeInfer(ShapeInfo* shape);
};

class Padding : public Transform {
 public:
  virtual void Init(const YAML::Node& item) {
    if (item["stride"].IsDefined()) {
      stride_ = item["stride"].as<int>();
      if (stride_ < 1) {
        std::cerr << "[Padding] coarest_stride should greater than 0"
                  << std::endl;
        exit(-1);
      }
    }
    if (item["width"].IsDefined() && item["height"].IsDefined()) {
      width_ = item["width"].as<int>();
      height_ = item["height"].as<int>();
    }
    if (item["im_padding_value"].IsDefined()) {
      im_value_ = item["im_padding_value"].as<std::vector<float>>();
    } else {
      im_value_ = {0, 0, 0};
    }
  }
  virtual bool Run(std::vector<cv::Mat> *ims);
  virtual bool ShapeInfer(ShapeInfo* shape);
  virtual void GeneralPadding(cv::Mat* im,
                              const std::vector<float> &padding_val,
                              int padding_w, int padding_h);
  virtual void MultichannelPadding(cv::Mat* im,
                                   const std::vector<float> &padding_val,
                                   int padding_w, int padding_h);
  virtual bool Run(std::vector<cv::Mat> *ims, int padding_w, int padding_h);
 private:
  int stride_ = -1;
  int width_ = 0;
  int height_ = 0;
  std::vector<float> im_value_;
};

class CenterCrop : public Transform {
 public:
  virtual void Init(const YAML::Node& item) {
    height_ = item["width"].as<int>();
    width_ = item["height"].as<int>();
  }
  virtual bool Run(std::vector<cv::Mat> *ims);
  virtual bool ShapeInfer(ShapeInfo* shape);

 private:
  int height_;
  int width_;
};


class Clip : public Transform {
 public:
  virtual void Init(const YAML::Node& item) {
    min_val_ = item["min_val"].as<std::vector<float>>();
    max_val_ = item["max_val"].as<std::vector<float>>();
  }

  virtual bool Run(std::vector<cv::Mat> *ims);
  virtual bool ShapeInfer(ShapeInfo* shape);

 private:
  std::vector<float> min_val_;
  std::vector<float> max_val_;
};


class Permute : public Transform {
 public:
  virtual void Init(const YAML::Node& item) {}
  virtual bool Run(std::vector<cv::Mat> *ims);
  virtual bool ShapeInfer(ShapeInfo* shape);

};

class Convert : pubulic Transform {
  public:
    virtual void Init(const YAML::Node& item) {
      dtype_ = item["dtype"].as<std::string>();
    }
    virtual bool Run(std::vector<cv::Mat> *ims);
    virtual bool ShapeInfer(ShapeInfo* shape);
  private:
    std::string dtype_;
}

}//namespace