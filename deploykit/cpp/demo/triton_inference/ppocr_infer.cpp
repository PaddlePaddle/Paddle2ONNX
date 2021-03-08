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
#include "include/deploy/postprocess/ppocr_post_proc.h"
#include "include/deploy/preprocess/ppocr_pre_proc.h"

DEFINE_string(model_name, "", "Path of inference model");
DEFINE_string(url, "", "url of triton server");
DEFINE_string(model_version, "", "model version of triton server");
DEFINE_string(cfg_file, "", "Path of yaml file");
DEFINE_string(image, "", "Path of test image file");
DEFINE_string(image_list, "", "Path of test image list file");
DEFINE_int32(batch_size, 1, "Batch size of infering");
DEFINE_string(toolkit, "ocr", "Type of PaddleToolKit");

int main(int argc, char **argv) {
  // Parsing command-line
  google::ParseCommandLineFlags(&argc, &argv, true);
  // parser yaml file
  Deploy::ConfigParser parser;
  parser.Load(FLAGS_cfg_file, FLAGS_toolkit);

  // data preprocess
  // preprocess init
  Deploy::PaddleOcrPreProc preprocess;
  preprocess.Init(parser);
  // postprocess init
  Deploy::PaddleOcrPostProc postprocess;
  postprocess.Init(parser);
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
    // Mini-batch predict
    std::string image_path;
    std::vector<std::string> image_paths;
    while (getline(inf, image_path)) {
      image_paths.push_back(image_path);
    }
    imgs = image_paths.size();
    for (int i = 0; i < image_paths.size(); i += FLAGS_batch_size) {
      int im_vec_size =
          std::min(static_cast<int>(image_paths.size()), i + FLAGS_batch_size);
      std::vector<cv::Mat> im_vec(im_vec_size - i);
      for (int j = i; j < im_vec_size; ++j) {
        im_vec[j - i] = std::move(cv::imread(image_paths[j], 1));
      }
      std::vector<Deploy::ShapeInfo> shape_traces;
      std::vector<Deploy::DataBlob> inputs;
      // preprocess
      preprocess.Run(im_vec, &inputs, &shape_traces);
      // infer
      std::vector<Deploy::DataBlob> outputs;
      triton_engine.Infer(configs, inputs, &outputs);
      // postprocess
      std::vector<Deploy::PaddleOcrResult> results;
      postprocess.Run(outputs, shape_traces, &results);
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
    preprocess.Run(imgs, &inputs, &shape_traces);
    // infer
    std::vector<Deploy::DataBlob> outputs;
    triton_engine.Infer(configs, inputs, &outputs);
    // postprocess
    std::vector<Deploy::PaddleOcrResult> results;
    postprocess.Run(outputs, shape_traces, &results);
  }
}
