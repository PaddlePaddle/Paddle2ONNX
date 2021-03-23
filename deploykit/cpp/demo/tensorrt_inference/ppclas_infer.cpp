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

#include <fstream>
#include <iostream>

#include "yaml-cpp/yaml.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "include/deploy/engine/tensorrt_engine.h"
#include "include/deploy/postprocess/ppclas_post_proc.h"
#include "include/deploy/preprocess/ppclas_pre_proc.h"

DEFINE_string(model_dir, "", "Path of inference model");
DEFINE_string(cfg_file, "", "Path of yaml file");
DEFINE_string(image, "", "Path of test image file");
DEFINE_string(image_list, "", "Path of test image list file");
DEFINE_string(trt_cache_file, "", "Cache path to store optimized trt file");
DEFINE_int32(batch_size, 1, "Batch size of infering");
DEFINE_string(toolkit, "clas", "Type of PaddleToolKit");

int main(int argc, char **argv) {
  // Parsing command-line
  google::ParseCommandLineFlags(&argc, &argv, true);
  // parser yaml file
  Deploy::ConfigParser parser;
  parser.Load(FLAGS_cfg_file, FLAGS_toolkit);

  // data preprocess
  // preprocess init
  Deploy::PaddleClasPreProc preprocess;
  preprocess.Init(parser);
  // postprocess init
  Deploy::PaddleClasPostProc postprocess;
  postprocess.Init(parser);
  // engine init
  Deploy::TensorRTInferenceEngine tensorrt_engine;

  Deploy::TensorRTInferenceConfigs configs;
  configs.SetTRTDynamicShapeInfo("inputs",
          {1, 3, 224, 224},
          {1, 3, 224, 224},
          {1, 3, 224, 224});
  tensorrt_engine.Init(FLAGS_model_dir,
          1<<28,
          1,
          FLAGS_trt_cache_file,
          configs);

  // read image
  std::vector<cv::Mat> imgs;
  cv::Mat img;
  img = cv::imread(FLAGS_image, 1);
  imgs.push_back(std::move(img));
  // create inpus and shape_traces
  std::vector<Deploy::DataBlob> inputs;
  // preprocess
  preprocess.Run(imgs, &inputs);
  // infer
  std::vector<Deploy::DataBlob> outputs;
  tensorrt_engine.Infer(inputs, 0, &outputs);
  // postprocess
  std::vector<Deploy::PaddleClasResult> results;
  postprocess.Run(outputs, &results);
  // print result
  Deploy::PaddleClasResult result = results[0];
  std::cout << "result: " << std::endl;
  std::cout << "\tclass id: " << result.class_id << std::endl;
  std::cout << "\tscore: " << result.score << std::endl;
}
