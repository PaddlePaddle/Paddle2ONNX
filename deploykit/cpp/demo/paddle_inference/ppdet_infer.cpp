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

#include "yaml-cpp/yaml.h"
#include "include/deploy/common/config.h"
#include "include/deploy/common/blob.h"
#include "include/deploy/engine/engine_config.h"
#include "include/deploy/postprocess/ppdet_post_proc.h"
#include "include/deploy/preprocess/ppdet_pre_proc.h"
#include "include/deploy/engine/ppinference_engine.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>


DEFINE_string(model_dir, "", "Path of inference model");
DEFINE_string(cfg_file, "", "Path of yaml file");
DEFINE_string(image, "", "Path of test image file");
DEFINE_string(image_list, "", "Path of test image list file");
DEFINE_bool(use_gpu, false, "Infering with GPU or CPU");
DEFINE_int32(gpu_id, 0, "GPU card id");
DEFINE_int32(batch_size, 1, "Batch size of infering");
DEFINE_string(pptype, "det", "Type of PaddleToolKit");


int main(int argc, char** argv) {
    // Parsing command-line
    google::ParseCommandLineFlags(&argc, &argv, true);
    //parser yaml file
    Deploy::ConfigParser parser;
    parser.Load(FLAGS_cfg_file, FLAGS_pptype);

    // data preprocess
    // preprocess init
    Deploy::PaddleDetPreProc detpreprocess;
    detpreprocess.Init(parser);
    // postprocess init
    Deploy::PaddleDetPostProc detpostprocess;
    detpostprocess.Init(parser);
    //engine init
    Deploy::PaddleInferenceEngine ppi_engine;
    Deploy::PaddleInferenceConfig ppi_config;
    ppi_engine.Init(FLAGS_model_dir, ppi_config);
    if (FLAGS_image_list != "") {
        //img_list
    } else {
        //read image
        std::vector<cv::Mat> imgs;
        cv::Mat img;
        img = cv::imread(FLAGS_image, 1);
        imgs.push_back(std::move(img));
        //create inpus and shape_traces
        std::vector<Deploy::ShapeInfo> shape_traces;
        std::vector<Deploy::DataBlob> inputs;
        //preprocess 
        detpreprocess.Run(imgs, &inputs, &shape_traces);
        //infer
        std::vector<Deploy::DataBlob> outputs;
        ppi_engine.Infer(inputs, &outputs);
        //postprocess
        std::vector<Deploy::PaddleDetResult> detresults;
        detpostprocess.Run(outputs, shape_traces, &detresults);
    }
}
