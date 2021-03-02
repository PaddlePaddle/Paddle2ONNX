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

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "yaml-cpp/yaml.h"
#include "include/deploy/postprocess/ppocr_post_proc.h"
#include "include/deploy/preprocess/ppocr_pre_proc.h"
#include "include/deploy/engine/ppinference_engine.h"



DEFINE_string(det_model_dir, "", "Path of det inference model");
DEFINE_string(det_cfg_file, "", "Path of det yaml file");
DEFINE_string(image, "", "Path of test image file");
DEFINE_string(toolkit, "ocr", "Type of PaddleToolKit");


int main(int argc, char** argv) {
  // Parsing command-line
  google::ParseCommandLineFlags(&argc, &argv, true);

  // init det model
  // det parser yaml file
  Deploy::ConfigParser det_parser;
  det_parser.Load(FLAGS_det_cfg_file, FLAGS_toolkit);
  // det preprocess init
  Deploy::PaddleOcrPreProc det_preprocess;
  det_preprocess.Init(det_parser);
  // det postprocess init
  Deploy::PaddleOcrPostProc det_postprocess;
  det_postprocess.Init(det_parser);
  // engine init
  Deploy::PaddleInferenceEngine det_ppi_engine;
  Deploy::PaddleInferenceConfig det_ppi_config;
  det_ppi_engine.Init(FLAGS_det_model_dir, det_ppi_config);
  // read image
  std::vector<cv::Mat> imgs;
  cv::Mat img;
  img = cv::imread(FLAGS_image, 1);
  imgs.push_back(std::move(img));
  // create inpus and shape_infos
  std::vector<Deploy::ShapeInfo> shape_infos;
  std::vector<Deploy::DataBlob> inputs;
  // preprocess
  det_preprocess.Run(imgs, &inputs, &shape_infos);
  // det infer
  std::vector<Deploy::DataBlob> outputs;
  det_ppi_engine.Infer(inputs, &outputs);
  // postprocess
  std::vector<Deploy::PaddleOcrResult> ocr_results;
  det_postprocess.Run(outputs, shape_infos, &ocr_results);
}

