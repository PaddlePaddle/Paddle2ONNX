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

#include "include/deploy/postprocess/ppdet_post_proc.h"

namespace Deploy {

void PaddleDetPostProc::Init(const ConfigParser &parser) {
  model_arch_ = parser.Get<std::string>("model_name");
  labels_.clear();
  int i = 0;
  for (auto item : parser.GetNode("labels")) {
    std::string label = item.as<std::string>();
    labels_[i] = label;
    i++;
  }
}

bool PaddleDetPostProc::Run(const std::vector<DataBlob> &outputs,
                            const std::vector<ShapeInfo> &shape_traces,
                            std::vector<PaddleDetResult> *det_results) {
  det_results->clear();
  DataBlob output_blob = outputs[0];
  float *output_data = reinterpret_cast<float*>(output_blob.data.data());
  auto lod_vector = output_blob.lod;
  int batchsize = shape_traces.size();
  // box postprocess
  for (int i = 0; i < lod_vector[0].size() - 1; ++i) {
    int rh = 1;
    int rw = 1;
    if (model_arch_ == "SSD" || model_arch_ == "Face") {
      rh =  shape_traces[i].shape[0][1];
      rw =  shape_traces[i].shape[0][0];
    }
    PaddleDetResult det_result;
    for (int j = lod_vector[0][i]; j < lod_vector[0][i + 1]; ++j) {
      Box box;
      box.category_id = static_cast<int>(round(output_data[j * 6]));
      box.category = labels_[box.category_id];
      box.score = output_data[1 + j * 6];
      int xmin = (output_data[2 + j * 6] * rw);
      int ymin = (output_data[3 + j * 6] * rh);
      int xmax = (output_data[4 + j * 6] * rw);
      int ymax = (output_data[5 + j * 6] * rh);
      int wd = xmax - xmin;
      int hd = ymax - ymin;
      box.coordinate = {xmin, ymin, wd, hd};
      det_result.boxes.push_back(std::move(box));
    }
    det_results->push_back(std::move(det_result));
  }
}

}  // namespace Deploy
