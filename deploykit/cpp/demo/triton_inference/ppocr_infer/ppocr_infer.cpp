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

DEFINE_string(cls_model_name, "", "Path of inference model");
DEFINE_string(rec_model_name, "", "Path of inference model");
DEFINE_string(det_model_name, "", "Path of inference model");
DEFINE_string(url, "", "url of triton server");
DEFINE_string(model_version, "", "model version of triton server");
DEFINE_string(det_cfg_file, "", "Path of det yaml file");
DEFINE_string(cls_cfg_file, "", "Path of cls yaml file");
DEFINE_string(rec_cfg_file, "", "Path of rec yaml file");
DEFINE_string(image, "", "Path of test image file");
DEFINE_string(toolkit, "ocr", "Type of PaddleToolKit");

int main(int argc, char **argv) {
  // Parsing command-line
  google::ParseCommandLineFlags(&argc, &argv, true);

  // engine init
  Deploy::TritonInferenceEngine triton_engine;
  triton_engine.Init(FLAGS_url);

  // detect init
  Deploy::ConfigParser det_parser;
  det_parser.Load(FLAGS_det_cfg_file, FLAGS_toolkit);
  Deploy::PaddleOcrPreProc det_preprocess;
  det_preprocess.Init(det_parser);
  Deploy::PaddleOcrPostProc det_postprocess;
  det_postprocess.Init(det_parser);

  Deploy::TritonInferenceConfigs det_configs(FLAGS_det_model_name);
  det_configs.model_version_ = FLAGS_model_version;


  // read image
  std::vector<cv::Mat> imgs;
  cv::Mat img;
  img = cv::imread(FLAGS_image, 1);
  imgs.push_back(std::move(img));
  // create inpus and shape_traces
  std::vector<Deploy::ShapeInfo> shape_traces;
  std::vector<Deploy::DataBlob> inputs;
  // preprocess
  det_preprocess.Run(imgs, &inputs, &shape_traces);
  // infer
  std::vector<Deploy::DataBlob> outputs;
  triton_engine.Infer(det_configs, inputs, &outputs);
  // postprocess
  std::vector<Deploy::PaddleOcrResult> results;
  det_postprocess.Run(outputs, shape_traces, &results);


  // cls init
  Deploy::ConfigParser cls_parser;
  cls_parser.Load(FLAGS_cls_cfg_file, FLAGS_toolkit);
  Deploy::PaddleOcrPreProc cls_preprocess;
  cls_preprocess.Init(cls_parser);
  Deploy::PaddleOcrPostProc cls_postprocess;
  cls_postprocess.Init(cls_parser);
  Deploy::TritonInferenceConfigs cls_configs(FLAGS_cls_model_name);
  cls_configs.model_version_ = FLAGS_model_version;

  // rec init
  Deploy::ConfigParser rec_parser;
  rec_parser.Load(FLAGS_rec_cfg_file, FLAGS_toolkit);
  Deploy::PaddleOcrPreProc rec_preprocess;
  rec_preprocess.Init(rec_parser);
  Deploy::PaddleOcrPostProc rec_postprocess;
  rec_postprocess.Init(rec_parser);
  Deploy::TritonInferenceConfigs rec_configs(FLAGS_rec_model_name);
  rec_configs.model_version_ = FLAGS_model_version;

  // do ocr
  int img_num = results.size();
  for (int i = 0; i <img_num; i++) {
    // crop image from det boxes
    std::vector<std::vector<std::vector<int>>> boxes;
    std::vector <cv::Mat> crop_imgs;
    boxes = results[i].boxes;
    cv::Mat src_img = imgs[i];
    det_postprocess.GetRotateCropImage(src_img, boxes, &crop_imgs);
    std::vector<Deploy::PaddleOcrResult> results;
    // run cls model
    if (FLAGS_cls_model_name != "") {
      std::vector<Deploy::ShapeInfo> cls_shape_infos;
      std::vector<Deploy::DataBlob> cls_inputs;
      cls_preprocess.Run(crop_imgs, &cls_inputs, &cls_shape_infos);
      std::vector<Deploy::DataBlob> cls_outputs;

      triton_engine.Infer(cls_configs, cls_inputs, &cls_outputs);
      cls_postprocess.Run(cls_outputs, cls_shape_infos, &results);
      for (int j = 0; j < crop_imgs.size(); j ++) {
        if (results[j].label % 2 == 1 &&
            results[j].cls_score > cls_parser.Get<double>("cls_thresh")) {
          cv::rotate(crop_imgs[j], crop_imgs[j], 1);
        }
      }
    }
    // run rec model
    for (int j = 0; j < crop_imgs.size(); j++) {
      std::vector<Deploy::ShapeInfo> rec_shape_infos;
      std::vector<Deploy::DataBlob> rec_inputs;
      std::vector<cv::Mat> rec_imgs;
      rec_imgs.push_back(crop_imgs[j]);
      rec_preprocess.Run(rec_imgs, &rec_inputs, &rec_shape_infos);
      std::vector<Deploy::DataBlob> rec_outputs;
      triton_engine.Infer(rec_configs, rec_inputs, &rec_outputs);
      rec_postprocess.Run(rec_outputs, rec_shape_infos, &results);
      std::vector<std::string> str_res = results[0].str_res;
      for (int k = 0; k < str_res.size(); k++) {
        std::cout << str_res[k];
      }
      std::cout << "\tscore: " << results[0].crnn_score << std::endl;
    }
  }
}
