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
                            const std::vector<ShapeInfo> &shape_infos,
                            const bool use_cpu_nms,
                            std::vector<PaddleDetResult> *det_results) {
  det_results->clear();
  if (use_cpu_nms) {
    DetPostWithNms(outputs, shape_infos, det_results);
  } else {
    DetPostNonNms(outputs, shape_infos, det_results);
  }
}

bool PaddleDetPostProc::DetPostWithNms(const std::vector<DataBlob> &outputs,
                                  const std::vector<ShapeInfo> &shape_infos,
                                  std::vector<PaddleDetResult> *det_results) {
  int keep_top_k = 100;
  DataBlob box_blob = outputs[0];
  DataBlob score_blob = outputs[1];
  std::vector<int> box_shape = box_blob.shape;
  std::vector<int> score_shape = score_blob.shape;
  int batchsize = score_shape[0];
  int cls_num = score_shape[1];
  int nms_box = score_shape[2];
  int score_size = cls_num * nms_box;
  int box_size = box_shape[1] * box_shape[2];
  for (int i = 0; i < batchsize; i++) {
    std::map<int, std::vector<int>> indices;
    int num_det = 0;
    for (int j = 0; j < cls_num; j++) {
      NMSFast(score_blob, box_blob, i, j, &(indices[j]));
      num_det += indices[j].size();
    }
    float *scores_data =
      reinterpret_cast<float*>(score_blob.data.data()) + i * score_size;
    float *box_data =
      reinterpret_cast<float*>(score_blob.data.data()) + i * box_size;
    if (keep_top_k > -1 && num_det > keep_top_k) {
      float *sdata;
      std::vector<std::pair<float, std::pair<int, int>>> score_index_pairs;
      for (const auto &it : indices) {
        int label = it.first;
        sdata = scores_data + label * nms_box;
        const std::vector<int>& label_indices = it.second;
        for (size_t j = 0; j < label_indices.size(); ++j) {
          int idx = label_indices[j];
          score_index_pairs.push_back(
              std::make_pair(sdata[idx], std::make_pair(label, idx)));
        }
      }
      std::stable_sort(score_index_pairs.begin(), score_index_pairs.end(),
                      SortScorePairDescend<std::pair<int, int>>);
      score_index_pairs.resize(keep_top_k);
      std::map<int, std::vector<int>> new_indices;
      for (size_t j = 0; j < score_index_pairs.size(); ++j) {
          int label = score_index_pairs[j].second.first;
          int idx = score_index_pairs[j].second.second;
          new_indices[label].push_back(idx);
      }
      indices.clear();
      new_indices.swap(indices);
    }
    int rh = 1;
    int rw = 1;
    if (model_arch_ == "SSD" || model_arch_ == "Face") {
      rh =  shape_infos[i].shape[0][1];
      rw =  shape_infos[i].shape[0][0];
    }
    PaddleDetResult det_result;
    for (const auto& it : indices) {
      int label = it.first;
      const std::vector<int>& label_indices = it.second;
      for (size_t j = 0; j < label_indices.size(); ++j) {
        Box box;
        int idx = label_indices[j];
        box.category_id = label;
        box.category = labels_[box.category_id];
        box.score = scores_data[label * nms_box + idx];
        float xmin = box_data[4 * idx] * rw;
        float ymin = box_data[4 * idx] * rh;
        float xmax = box_data[4 * idx] * rw;
        float ymax = box_data[4 * idx] * rh;
        float wd = xmax - xmin;
        float hd = ymax - ymin;
        box.coordinate = {xmin, ymin, wd, hd};
        det_result.boxes.push_back(std::move(box));
      }
    }
    det_results->push_back(std::move(det_result));
  }
}

bool PaddleDetPostProc::DetPostNonMms(const std::vector<DataBlob> &outputs,
                                const std::vector<ShapeInfo> &shape_infos,
                                std::vector<PaddleDetResult> *det_results) {
  DataBlob output_blob = outputs[0];
  float *output_data = reinterpret_cast<float*>(output_blob.data.data());
  if (output_blob.lod.empty()) {
    std::vector<size_t> lod = {0, output_blob.data.size()/sizeof(float)};
    output_blob.lod.push_back(lod);
  }
  auto lod_vector = output_blob.lod;
  // box postprocess
  for (int i = 0; i < lod_vector[0].size() - 1; ++i) {
    int rh = 1;
    int rw = 1;
    if (model_arch_ == "SSD" || model_arch_ == "Face") {
      rh =  shape_infos[i].shape[0][1];
      rw =  shape_infos[i].shape[0][0];
    }
    PaddleDetResult det_result;
    for (int j = lod_vector[0][i]; j < lod_vector[0][i + 1]; ++j) {
      Box box;
      box.category_id = static_cast<int>(round(output_data[j * 6]));
      box.category = labels_[box.category_id];
      box.score = output_data[1 + j * 6];
      float xmin = (output_data[2 + j * 6] * rw);
      float ymin = (output_data[3 + j * 6] * rh);
      float xmax = (output_data[4 + j * 6] * rw);
      float ymax = (output_data[5 + j * 6] * rh);
      float wd = xmax - xmin;
      float hd = ymax - ymin;
      box.coordinate = {xmin, ymin, wd, hd};
      det_result.boxes.push_back(std::move(box));
    }
    det_results->push_back(std::move(det_result));
  }
  if (outputs.size() == 2) {
    DataBlob mask_blob = outputs[0];
    std::vector<int> output_mask_shape = mask_blob.shape;
    float *mask_data = reinterpret_cast<float*>(mask_blob.data.data());
    int masks_size = 1;
    for (const auto& i : output_mask_shape) {
      masks_size *= i;
    }
    int mask_pixels = output_mask_shape[2] * output_mask_shape[3];
    int classes = output_mask_shape[1];
    int mask_idx = 0;
    for (int i = 0; i < lod_vector[0].size() - 1; ++i) {
      (*det_results)[i].mask_resolution = output_mask_shape[2];
      for (int j = 0; j < (*det_results)[i].boxes.size(); ++j) {
        Box *box = &(*det_results)[i].boxes[i];
        int category_id = box->category_id;
        box->mask.shape = {static_cast<int>(box->coordinate[2]),
                        static_cast<int>(box->coordinate[3])};
        auto begin_mask =
        mask_data + (i * classes + box->category_id) * mask_pixels;
        cv::Mat bin_mask(output_mask_shape[2],
                        output_mask_shape[2],
                        CV_32FC1,
                        begin_mask);
        cv::resize(bin_mask, bin_mask, cv::Size(box->mask.shape[0],
                  box->mask.shape[1]));
          cv::threshold(bin_mask, bin_mask, 0.5, 1, cv::THRESH_BINARY);
        auto mask_int_begin = reinterpret_cast<float*>(bin_mask.data);
        auto mask_int_end =
          mask_int_begin + box->mask.shape[0] * box->mask.shape[1];
        box->mask.data.assign(mask_int_begin, mask_int_end);
        mask_idx++;
      }
    }
  }
}

void PaddleDetPostProc::NMSFast(const DataBlob &score_blob,
                          const DataBlob &box_blob,
                          const int &i, const int &j,
                          std::vector<int> *selected_indices) {
  const float score_threshold = 0.009999999776482582;
  const int top_k = 100;
  float nms_threshold = 0.44999998807907104;
  bool normalized = false;
  std::vector<int> box_shape = box_blob.shape;
  int size_b = 0;
  for (int i = 0; i < box_shape.size(); i++) {
    size_b *= box_shape[i];
  }
  int box_n = box_shape[1];
  std::vector<int> score_shape = score_blob.shape;
  int size_s = 0;
  for (int i = 0; i < score_shape.size(); i++) {
    size_s *= score_shape[i];
  }
  float *score =
    reinterpret_cast<float*>(score_blob.data.data()) + i * size_s + j * box_n;
  float *bbox_data =
    reinterpret_cast<float*>(box_blob.data.data()) + i * size_b;
  std::vector<float> scores_data(box_n);
  std::copy_n(score, box_n, scores_data.begin());
  std::vector<std::pair<float, int>> sorted_indices;
  GetMaxScoreIndex(scores_data, score_threshold, top_k, &sorted_indices);
  selected_indices->clear();
  while (sorted_indices.size() != 0) {
    const int idx = sorted_indices.front().second;
    bool keep = true;
    for (size_t k = 0; k < selected_indices->size(); ++k) {
      if (keep) {
        const int kept_idx = (*selected_indices)[k];
        float overlap = 0.0f;
        overlap = JaccardOverlap(bbox_data + idx * 4,
                                  bbox_data + kept_idx * 4, normalized);
        keep = overlap <= nms_threshold;
      } else {
        break;
      }
    }
    if (keep) {
      selected_indices->push_back(idx);
    }
    sorted_indices.erase(sorted_indices.begin());
  }
}

void PaddleDetPostProc::GetMaxScoreIndex(const std::vector<float> &scores,
                          const float &threshold, const int &top_k,
                          std::vector<std::pair<float, int>> *sorted_indices) {
  for (size_t i = 0; i < scores.size(); ++i) {
    if (scores[i] > threshold) {
      sorted_indices->push_back(std::make_pair(scores[i], i));
    }
  }
  // Sort the score pair according to the scores in descending order
  std::stable_sort(sorted_indices->begin(), sorted_indices->end(),
                   SortScorePairDescend<int>);
  // Keep top_k scores if needed.
  if (top_k > -1 && top_k < static_cast<int>(sorted_indices->size())) {
    sorted_indices->resize(top_k);
  }
}

float PaddleDetPostProc::JaccardOverlap(const float* box1,
                                const float* box2,
                                const bool normalized) {
  if (box2[0] > box1[2] || box2[2] < box1[0] || box2[1] > box1[3] ||
      box2[3] < box1[1]) {
    return static_cast<float>(0.);
  } else {
    const float inter_xmin = std::max(box1[0], box2[0]);
    const float inter_ymin = std::max(box1[1], box2[1]);
    const float inter_xmax = std::min(box1[2], box2[2]);
    const float inter_ymax = std::min(box1[3], box2[3]);
    float norm = normalized ? static_cast<float>(0.) : static_cast<float>(1.);
    float inter_w = inter_xmax - inter_xmin + norm;
    float inter_h = inter_ymax - inter_ymin + norm;
    const float inter_area = inter_w * inter_h;
    const float bbox1_area = BBoxArea(box1, normalized);
    const float bbox2_area = BBoxArea(box2, normalized);
    return inter_area / (bbox1_area + bbox2_area - inter_area);
  }
}


float PaddleDetPostProc::BBoxArea(const float* box, const bool normalized) {
  if (box[2] < box[0] || box[3] < box[1]) {
    // If coordinate values are is invalid
    // (e.g. xmax < xmin or ymax < ymin), return 0.
    return static_cast<float>(0.);
  } else {
    const float w = box[2] - box[0];
    const float h = box[3] - box[1];
    if (normalized) {
      return w * h;
    } else {
      // If coordinate values are not within range [0, 1].
      return (w + 1) * (h + 1);
    }
  }
}

}  // namespace Deploy
