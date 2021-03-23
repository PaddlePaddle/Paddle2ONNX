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

#include "include/deploy/postprocess/ppseg_post_proc.h"


namespace Deploy {

bool PaddleSegPostProc::Run(const std::vector<DataBlob> &outputs,
                            const std::vector<ShapeInfo> &shape_infos,
                            std::vector<PaddleSegResult> *seg_results) {
  DataBlob data_blob = outputs[0];
  std::vector<int> shape = data_blob.shape;
  int batchsize = shape[0];
  float *data = reinterpret_cast<float*>(data_blob.data.data());
  int size = 0;
  for (int i = 1; i < shape.size(); i++) {
    size *= shape[i];
  }
  for (int i = 0; i < batchsize; i++) {
    PaddleSegResult seg_result;
    std::vector<uint8_t> label;
    std::vector<float> score;
    std::vector<int> final_shape = shape_infos[i].shape.back();
    std::vector<int> origin_shape = shape_infos[i].shape[0];
    int output_size = final_shape[0] * final_shape[1];
    label.resize(output_size);
    score.resize(output_size);
    for (int j = 0; j < output_size; j++) {
      std::vector<float> pixel_score;
      pixel_score.resize(shape[1]);
      for (int k = 0; k < shape[1]; k++) {
        pixel_score[k] = data[j + k * output_size];
      }
      int index = std::max_element(
        pixel_score.begin(), pixel_score.end()) - pixel_score.begin();
      label[j] = static_cast<uint8_t>(index);
      score[j] = pixel_score[index];
    }
    cv::Mat mask_label(final_shape[1], final_shape[0], CV_8UC1, label.data());
    cv::Mat mask_score(final_shape[1], final_shape[0], CV_32FC1, score.data());
    for (int j = 0; j < shape_infos[i].transform_order.size(); j++) {
      std::string name = shape_infos[i].transform_order[j];
      if (name == "Padding") {
        int before_pad_w = shape_infos[i].shape[j][0];
        int before_pad_h = shape_infos[i].shape[j][1];
        mask_label = mask_label(cv::Rect(0, 0, before_pad_w, before_pad_h));
        mask_score = mask_score(cv::Rect(0, 0, before_pad_w, before_pad_h));
      } else if (name.find("Resize", 0) != -1) {
        int before_resize_w = shape_infos[i].shape[j][0];
        int before_resize_h = shape_infos[i].shape[j][1];
        cv::resize(mask_label,
                 mask_label,
                 cv::Size(before_resize_w, before_resize_h),
                 0,
                 0,
                 cv::INTER_NEAREST);
        cv::resize(mask_score,
                 mask_score,
                 cv::Size(before_resize_w, before_resize_h),
                 0,
                 0,
                 cv::INTER_LINEAR);
      }
    }
    seg_result.label_map.data.assign(mask_label.begin<uint8_t>(),
                                    mask_label.end<uint8_t>());
    seg_result.label_map.shape = {mask_label.rows, mask_label.cols};
    seg_result.score_map.data.assign(mask_score.begin<float>(),
                                mask_score.end<float>());
    seg_result.score_map.shape = {mask_score.rows, mask_score.cols};
    seg_results->push_back(std::move(seg_result));
  }
}

}  // namespace Deploy
