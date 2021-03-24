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

#include <gflags/gflags.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "yaml-cpp/yaml.h"
#include "include/deploy/postprocess/ppclas_post_proc.h"
#include "include/deploy/preprocess/ppclas_pre_proc.h"
#include "include/deploy/engine/openvino_engine.h"

DEFINE_string(model_filename, "", "Path of det inference model");
DEFINE_string(cfg_file, "", "Path of yaml file");
DEFINE_string(image, "", "Path of test image file");
DEFINE_string(device, "CPU", "Infering with VPU or CPU");
DEFINE_int32(batch_size, 1, "Batch size of infering");
DEFINE_string(toolkit, "clas", "Type of PaddleToolKit");

int main(int argc, char** argv) {
  // Parsing command-line
  google::ParseCommandLineFlags(&argc, &argv, true);
  // parser yaml file
  Deploy::ConfigParser parser;
  parser.Load(FLAGS_cfg_file, FLAGS_toolkit);
  // data preprocess
  // preprocess init
  Deploy::PaddleClasPreProc clas_preprocess;
  clas_preprocess.Init(parser);
  // postprocess init
  Deploy::PaddleClasPostProc clas_postprocess;
  clas_postprocess.Init(parser);
  // engine init
  Deploy::OpenVinoEngine openvino_engine;
  Deploy::OpenVinoEngineConfig openvino_config;
  openvino_config.device = FLAGS_device;
  openvino_engine.Init(FLAGS_model_filename, openvino_config);
  // read image
  std::vector<cv::Mat> imgs;
  cv::Mat img;
  img = cv::imread(FLAGS_image, 1);
  imgs.push_back(std::move(img));
  // create inpus
  std::vector<Deploy::DataBlob> inputs;
  // preprocess
  clas_preprocess.Run(imgs, &inputs);
  // infer
  std::vector<Deploy::DataBlob> outputs;
  openvino_engine.Infer(inputs, &outputs);
  // postprocess
  std::vector<Deploy::PaddleClasResult> clas_results;
  clas_postprocess.Run(outputs, &clas_results);
  // print result
  Deploy::PaddleClasResult result = clas_results[0];
  std::cout << "result: " << std::endl;
  std::cout << "\tclass id: " << result.class_id << std::endl;
  std::cout << "\tscore: " << result.score << std::endl;
}
