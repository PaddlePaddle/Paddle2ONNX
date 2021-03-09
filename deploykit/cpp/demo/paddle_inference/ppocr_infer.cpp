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



DEFINE_string(det_model_filename, "", "Path of det inference model");
DEFINE_string(det_params_filename, "", "Path of det inference params");
DEFINE_string(cls_model_filename, "", "Path of cls inference model");
DEFINE_string(cls_params_filename, "", "Path of cls inference params");
DEFINE_string(crnn_model_filename, "", "Path of crnn inference model");
DEFINE_string(crnn_params_filename, "", "Path of crnn inference params");
DEFINE_string(det_cfg_file, "", "Path of det yaml file");
DEFINE_string(cls_cfg_file, "", "Path of cls yaml file");
DEFINE_string(crnn_cfg_file, "", "Path of crnn yaml file");
DEFINE_string(image, "", "Path of test image file");
DEFINE_string(toolkit, "ocr", "Type of PaddleToolKit");
DEFINE_bool(use_cls, false, "Whether to use cls model");


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
  det_ppi_engine.Init(FLAGS_det_model_filename,
                      FLAGS_det_params_filename,
                      det_ppi_config);
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

  // init cls
  Deploy::ConfigParser cls_parser;
  cls_parser.Load(FLAGS_cls_cfg_file, FLAGS_toolkit);
  Deploy::PaddleOcrPreProc cls_preprocess;
  cls_preprocess.Init(cls_parser);
  Deploy::PaddleOcrPostProc cls_postprocess;
  cls_postprocess.Init(cls_parser);
  Deploy::PaddleInferenceEngine cls_ppi_engine;
  Deploy::PaddleInferenceConfig cls_ppi_config;
  cls_ppi_engine.Init(FLAGS_cls_model_filename,
                      FLAGS_cls_params_filename,
                      cls_ppi_config);

  // init crnn
  Deploy::ConfigParser crnn_parser;
  crnn_parser.Load(FLAGS_crnn_cfg_file, FLAGS_toolkit);
  Deploy::PaddleOcrPreProc crnn_preprocess;
  crnn_preprocess.Init(crnn_parser);
  Deploy::PaddleOcrPostProc crnn_postprocess;
  crnn_postprocess.Init(crnn_parser);
  Deploy::PaddleInferenceEngine crnn_ppi_engine;
  Deploy::PaddleInferenceConfig crnn_ppi_config;
  crnn_ppi_engine.Init(FLAGS_crnn_model_filename,
                      FLAGS_crnn_params_filename,
                      crnn_ppi_config);

  // do ocr
  int img_num = ocr_results.size();
  for (int i = 0; i <img_num; i++) {
    // crop image from det boxes
    std::vector<std::vector<std::vector<int>>> boxes;
    std::vector <cv::Mat> crop_imgs;
    boxes = ocr_results[i].boxes;
    cv::Mat src_img = imgs[i];
    det_postprocess.GetRotateCropImage(src_img, boxes, &crop_imgs);
    std::vector<Deploy::PaddleOcrResult> results;
    // run cls model
    if (FLAGS_use_cls) {
      std::vector<Deploy::ShapeInfo> cls_shape_infos;
      std::vector<Deploy::DataBlob> cls_inputs;
      cls_preprocess.Run(crop_imgs, &cls_inputs, &cls_shape_infos);
      std::vector<Deploy::DataBlob> cls_outputs;
      cls_ppi_engine.Infer(cls_inputs, &cls_outputs);
      cls_postprocess.Run(cls_outputs, cls_shape_infos, &results);
      for (int j = 0; j < crop_imgs.size(); j ++) {
        if (results[j].label % 2 == 1 &&
            results[j].score > results[j].cls_thresh) {
          cv::rotate(crop_imgs[j], crop_imgs[j], 1);
        }
      }
    }
    // run crnn model
    for (int j = 0; j < crop_imgs.size(); j++) {
      std::vector<Deploy::ShapeInfo> crnn_shape_infos;
      std::vector<Deploy::DataBlob> crnn_inputs;
      std::vector<cv::Mat> rec_imgs;
      rec_imgs.push_back(crop_imgs[j])
      crnn_preprocess.Run(rec_imgs, &crnn_inputs, &crnn_shape_infos);
      std::vector<Deploy::DataBlob> crnn_outputs;
      crnn_ppi_engine.Infer(crnn_inputs, &crnn_outputs);
      crnn_postprocess.Run(crnn_outputs, crnn_shape_infos, &results);
    }
  }
}

