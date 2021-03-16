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

#include <glog/logging.h>

#include <fstream>
#include <iostream>

#include "yaml-cpp/yaml.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "include/deploy/engine/triton_engine.h"
#include "include/deploy/postprocess/ppdet_post_proc.h"
#include "include/deploy/preprocess/ppdet_pre_proc.h"

DEFINE_string(model_name, "", "Path of inference model");
DEFINE_string(url, "", "url of triton server");
DEFINE_string(model_version, "", "model version of triton server");
DEFINE_string(cfg_file, "", "Path of yaml file");
DEFINE_string(image, "", "Path of test image file");
DEFINE_string(image_list, "", "Path of test image list file");
DEFINE_string(toolkit, "det", "Type of PaddleToolKit");

int main(int argc, char **argv) {
  // Parsing command-line
  google::ParseCommandLineFlags(&argc, &argv, true);
  // parser yaml file
  Deploy::ConfigParser parser;
  parser.Load(FLAGS_cfg_file, FLAGS_toolkit);

  // data preprocess
  // preprocess init
  Deploy::PaddleDetPreProc detpreprocess;
  detpreprocess.Init(parser);
  // postprocess init
  Deploy::PaddleDetPostProc detpostprocess;
  detpostprocess.Init(parser);
  // engine init
  Deploy::TritonInferenceEngine triton_engine;
  triton_engine.Init(FLAGS_url);

  Deploy::TritonInferenceConfigs configs(FLAGS_model_name);
  configs.model_version_ = FLAGS_model_version;

  int imgs = 1;
  if (FLAGS_image_list != "") {
    // img_list
    std::ifstream inf(FLAGS_image_list);
    if (!inf) {
      std::cerr << "Fail to open file " << FLAGS_image_list << std::endl;
      return -1;
    }
    // predict image list with batch_size == 1
    std::string image_path;
    std::vector<std::string> image_paths;
    while (getline(inf, image_path)) {
      image_paths.push_back(image_path);
    }
    imgs = image_paths.size();
    std::vector<cv::Mat> im_vec(1);
    for (int i = 0; i < image_paths.size(); i += 1) {
      im_vec[0] = std::move(cv::imread(image_paths[i], 1));
      std::vector<Deploy::ShapeInfo> shape_traces;
      std::vector<Deploy::DataBlob> inputs;
      // preprocess
      detpreprocess.Run(im_vec, &inputs, &shape_traces);
      // infer
      std::vector<Deploy::DataBlob> outputs;
      triton_engine.Infer(configs, inputs, &outputs);
      // postprocess
      std::vector<Deploy::PaddleDetResult> det_results;
      detpostprocess.Run(outputs, shape_traces, &det_results);

      for (int i = 0; i < det_results.size(); i++) {
        auto res = det_results[i];
        for (int j = 0; j < res.boxes.size(); j++) {
          auto box = res.boxes[j];

        if (box.score < 0.5 || box.score > 1.0) {
          continue;
        }
          std::string result_log;
          result_log += "class_id: " + std::to_string(box.category_id) + ", ";
          result_log += "score: " + std::to_string(box.score) + ", ";
          result_log += "coordinate[x:" + std::to_string(box.coordinate[0]);
          result_log += ", y:" + std::to_string(box.coordinate[1]);
          result_log += ", w:" + std::to_string(box.coordinate[2]);
          result_log += ", h:" + std::to_string(box.coordinate[3]) + "]";
          std::cout << result_log << std::endl;
        }
      }
    }
  } else {
    // read image
    std::vector<cv::Mat> imgs;
    cv::Mat img;
    img = cv::imread(FLAGS_image, 1);
    imgs.push_back(std::move(img));
    // create inpus and shape_traces
    std::vector<Deploy::ShapeInfo> shape_traces;
    std::vector<Deploy::DataBlob> inputs;
    // preprocess
    detpreprocess.Run(imgs, &inputs, &shape_traces);
    // infer
    std::vector<Deploy::DataBlob> outputs;
    triton_engine.Infer(configs, inputs, &outputs);
    // postprocess
    std::vector<Deploy::PaddleDetResult> det_results;
    detpostprocess.Run(outputs, shape_traces, &det_results);

    for (int i = 0; i < det_results.size(); i++) {
      auto res = det_results[i];
      for (int j = 0; j < res.boxes.size(); j++) {
        auto box = res.boxes[j];
        if (box.score < 0.5 ||  box.score > 1.0) {
          continue;
        }
        std::string result_log;
        result_log += "class_id: " + std::to_string(box.category_id) + ", ";
        result_log += "score: " + std::to_string(box.score) + ", ";
        result_log += "coordinate[x:" + std::to_string(box.coordinate[0]);
        result_log += ", y:" + std::to_string(box.coordinate[1]);
        result_log += ", w:" + std::to_string(box.coordinate[2]);
        result_log += ", h:" + std::to_string(box.coordinate[3]) + "]";
        std::cout << result_log << std::endl;
      }
    }
  }
}
