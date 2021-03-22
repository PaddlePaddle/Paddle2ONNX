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
#include "include/deploy/postprocess/ppseg_post_proc.h"
#include "include/deploy/preprocess/ppseg_pre_proc.h"
#include "include/deploy/engine/ppinference_engine.h"

DEFINE_string(model_filename, "", "Path of det inference model");
DEFINE_string(params_filename, "", "Path of det inference params");
DEFINE_string(cfg_file, "", "Path of yaml file");
DEFINE_string(image, "", "Path of test image file");
DEFINE_bool(use_gpu, false, "Infering with GPU or CPU");
DEFINE_int32(gpu_id, 0, "GPU card id");
DEFINE_int32(batch_size, 1, "Batch size of infering");
DEFINE_string(toolkit, "seg", "Type of PaddleToolKit");

int main(int argc, char** argv) {
  // Parsing command-line
  google::ParseCommandLineFlags(&argc, &argv, true);
  // parser yaml file
  Deploy::ConfigParser parser;
  parser.Load(FLAGS_cfg_file, FLAGS_toolkit);
  // data preprocess
  // preprocess init
  Deploy::PaddleSegPreProc seg_preprocess;
  seg_preprocess.Init(parser);
  // postprocess init
  Deploy::PaddleSegPostProc seg_postprocess;
  seg_postprocess.Init(parser);
  // engine init
  Deploy::PaddleInferenceEngine ppi_engine;
  Deploy::PaddleInferenceConfig ppi_config;
  ppi_config.use_gpu = FLAGS_use_gpu;
  ppi_engine.Init(FLAGS_model_filename, FLAGS_params_filename, ppi_config);
  // read image
  std::vector<cv::Mat> imgs;
  cv::Mat img;
  img = cv::imread(FLAGS_image, 1);
  imgs.push_back(std::move(img));
  // create inpus
  std::vector<Deploy::DataBlob> inputs;
  // create shape_info
  std::vector<Deploy::ShapeInfo> shape_infos;
  // preprocess
  seg_preprocess.Run(imgs, &inputs, &shape_infos);
  // infer
  std::vector<Deploy::DataBlob> outputs;
  ppi_engine.Infer(inputs, &outputs);
  // postprocess
  std::vector<Deploy::PaddleSegResult> seg_results;
  seg_postprocess.Run(outputs, shape_infos, &seg_results);
}
