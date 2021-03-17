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
#include "include/deploy/postprocess/ppdet_post_proc.h"
#include "include/deploy/preprocess/ppdet_pre_proc.h"
#include "include/deploy/engine/ppinference_engine.h"



DEFINE_string(model_filename, "", "Path of det inference model");
DEFINE_string(params_filename, "", "Path of det inference params");
DEFINE_string(cfg_file, "", "Path of yaml file");
DEFINE_string(image, "", "Path of test image file");
DEFINE_bool(use_gpu, false, "Infering with GPU or CPU");
DEFINE_int32(gpu_id, 0, "GPU card id");
DEFINE_int32(batch_size, 1, "Batch size of infering");
DEFINE_string(toolkit, "det", "Type of PaddleToolKit");
DEFINE_bool(use_cpu_nms, false, "whether postprocess with NMS");

int main(int argc, char** argv) {
  // Parsing command-line
  google::ParseCommandLineFlags(&argc, &argv, true);
  // parser yaml file
  Deploy::ConfigParser parser;
  parser.Load(FLAGS_cfg_file, FLAGS_toolkit);
  // data preprocess
  // preprocess init
  Deploy::PaddleDetPreProc det_preprocess;
  det_preprocess.Init(parser);
  // postprocess init
  Deploy::PaddleDetPostProc det_postprocess;
  det_postprocess.Init(parser);
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
  // create inpus and shape_infos
  std::vector<Deploy::ShapeInfo> shape_infos;
  std::vector<Deploy::DataBlob> inputs;
  // preprocess
  det_preprocess.Run(imgs, &inputs, &shape_infos);
  // infer
  std::vector<Deploy::DataBlob> outputs;
  ppi_engine.Infer(inputs, &outputs);
  // postprocess
  std::vector<Deploy::PaddleDetResult> det_results;
  det_postprocess.Run(outputs, shape_infos, FLAGS_use_cpu_nms, &det_results);
  // print result
  Deploy::PaddleDetResult result = det_results[0];
  for (int i = 0; i < result.boxes.size(); i++) {
    if (result.boxes[i].score > 0.3) {
      std::cout << "score: " << result.boxes[i].score
              << ", box(xmin, ymin, w, h):(" << result.boxes[i].coordinate[0]
              << ", " << result.boxes[i].coordinate[1] << ", "
              << result.boxes[i].coordinate[2] << ", "
              << result.boxes[i].coordinate[3] << ")" << std::endl;
    }
  }
}
