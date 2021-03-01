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

#include "include/deploy/postprocess/ppocr_post_proc.h"

namespace Deploy {

void PaddleOcrPostProc::Init(const ConfigParser &parser) {
  model_arch_ = parser.Get<std::string>("model_name");
  det_db_thresh_ = parser.Get<std::string>("det_db_thresh");
  det_db_unclip_ratio_ = parser.Get<std::string>("det_db_unclip_ratio");
}

bool PaddleOcrPostProc::Run(const std::vector<DataBlob> &outputs,
                            const std::vector<ShapeInfo> &shape_infos,
                            std::vector<PaddleOcrResult> *ocr_results) {
  ocr_results->clear();
  if (model_arch_ == "DET") {
    DetPostProc(outputs, shape_infos, ocr_results);
  }
}

bool PaddleOcrPostProc::DetPostProc(const std::vector<DataBlob> &outputs,
                            const std::vector<ShapeInfo> &shape_infos,
                            std::vector<PaddleOcrResult> *ocr_results) {
  DataBlob output_blob = outputs[0];
  float *output_data = reinterpret_cast<float*>(output_blob.data.data());
  std::vector<int> output_shape = output_blob.shape;
  int batchsize = output_shape[0];
  int n2 = output_shape[2];
  int n3 = output_shape[3];
  int n = n2 * n3;
  for (int i = 0; i < batchsize; i++) {
    PaddleOcrResult ocr_result;
    ShapeInfo shape_info = shape_infos[i];
    std::vector<float> pred(n, 0.0);
    std::vector<unsigned char> cbuf(n, ' ');
    for (int j = 0; j < n; j++) {
      pred[j] = output_data[i * n + j];
      cbuf[j] = (unsigned char)((out_data[i * n + j]) * 255);
    }
    cv::Mat cbuf_map(n2, n3, CV_8UC1, (unsigned char *)cbuf.data());  // NOLINT
    cv::Mat pred_map(n2, n3, CV_32F, (float *)pred.data());  // NOLINT
    const double threshold = det_db_thresh_ * 255;
    const double maxvalue = 255;
    cv::Mat bit_map;
    cv::threshold(cbuf_map, bit_map, threshold, maxvalue, cv::THRESH_BINARY);
    cv::Mat dilation_map;
    cv::Mat dila_ele =
              cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2));
    cv::dilate(bit_map, dilation_map, dila_ele);
    BoxesFromBitmap(pred_map,
                    dilation_map,
                    det_db_thresh_,
                    det_db_unclip_ratio_,
                    &ocr_result);
    FilterTagDetRes(shape_info, &ocr_result);
  }
}

PaddleOcrPostProc::BoxesFromBitmap(
            const cv::Mat &pred,
            const cv::Mat &bitmap,
            const float &box_thresh,
            const float &det_db_unclip_ratio,
            PaddleOcrResult *ocr_result) {
  const int min_size = 3;
  const int max_candidates = 1000;
  std::vector<std::vector<cv::Point>> contours;
  std::vector<cv::Vec4i> hierarchy;
  cv::findContours(bitmap, contours, hierarchy, cv::RETR_LIST,
                   cv::CHAIN_APPROX_SIMPLE);
  int num_contours =
      contours.size() >= max_candidates ? max_candidates : contours.size();
  for (int _i = 0; _i < num_contours; _i++) {
    if (contours[_i].size() <= 2) {
      continue;
    }
    float ssid;
    cv::RotatedRect box = cv::minAreaRect(contours[_i]);
    auto array = GetMiniBoxes(box, ssid);
    auto box_for_unclip = array;
    // end get_mini_box
    if (ssid < min_size) {
      continue;
    }
    float score;
    score = BoxScoreFast(array, pred);
    if (score < box_thresh)
      continue;
    // start for unclip
    cv::RotatedRect points = UnClip(box_for_unclip, det_db_unclip_ratio);
    if (points.size.height < 1.001 && points.size.width < 1.001) {
      continue;
    }
    // end for unclip
    cv::RotatedRect clipbox = points;
    auto cliparray = GetMiniBoxes(clipbox, ssid);
    if (ssid < min_size + 2)
      continue;
    float width = static_cast<float>(bitmap.cols);
    float height = static_cast<float>(bitmap.rows);
    float dest_width = static_cast<float>(pred.cols);
    float dest_height = static_cast<float>(pred.rows);
    std::vector<std::vector<int>> intcliparray;
    for (int num_pt = 0; num_pt < 4; num_pt++) {
      std::vector<int> a{static_cast<int>(
            clampf(roundf(cliparray[num_pt][0] / width * dest_width),
            0, dest_width)), static_cast<int>(
            clampf(roundf(cliparray[num_pt][1] / height * dest_height),
            0, dest_height))};
      intcliparray.push_back(a);
    }
    ocr_result->boxes.push_back(intcliparray);
  }
}

PaddleOcrPostProc::FilterTagDetRes(const ShapeInfo &shape_info,
                                  PaddleOcrResult *ocr_result) {
  int ori_w = shape_info.shape[0][0];
  int ori_h = shape_info.shape[0][1];
  float resize_w = reinterpret_cast<float>(shape_info.shape[1][0]);
  float resize_h = reinterpret_cast<float>(shape_info.shape[1][1]);
  float ratio_w = resize_w / reinterpret_cast<float>(ori_w);
  float ratio_h = resize_h / reinterpret_cast<float>(ori_h);

  std::vector<std::vector<std::vector<int>>> root_points;
  std::vector<std::vector<std::vector<int>>> boxes;
  boxes = ocr_result->boxes;
  ocr_result->boxes.clear();
  for (int n = 0; n < ocr_result->boxes.size(); n++) {
    boxes[n] = OrderPointsClockwise(boxes[n]);
    for (int m = 0; m < boxes[0].size(); m++) {
      boxes[n][m][0] /= ratio_w;
      boxes[n][m][1] /= ratio_h;
      boxes[n][m][0] =
        static_cast<int>(_min(_max(boxes[n][m][0], 0), ori_w - 1));
      boxes[n][m][1] =
        static_cast<int>(_min(_max(boxes[n][m][1], 0), ori_h - 1));
    }
  }
  for (int n = 0; n < boxes.size(); n++) {
    int rect_width, rect_height;
    rect_width = static_cast<int>(
                    sqrt(pow(boxes[n][0][0] - boxes[n][1][0], 2) +
                    pow(boxes[n][0][1] - boxes[n][1][1], 2)));
    rect_height = static_cast<int>(
                    sqrt(pow(boxes[n][0][0] - boxes[n][3][0], 2) +
                    pow(boxes[n][0][1] - boxes[n][3][1], 2)));
    if (rect_width <= 4 || rect_height <= 4)
      continue;
    root_points.push_back(boxes[n]);
  }
  ocr_result->boxes.push_back(root_points);
}

std::vector<std::vector<int>>
PaddleOcrPostProc::OrderPointsClockwise(std::vector<std::vector<int>> *pts) {
  std::vector<std::vector<int>> box = pts;
  std::sort(box.begin(), box.end(), XsortInt);

  std::vector<std::vector<int>> leftmost = {box[0], box[1]};
  std::vector<std::vector<int>> rightmost = {box[2], box[3]};

  if (leftmost[0][1] > leftmost[1][1])
    std::swap(leftmost[0], leftmost[1]);

  if (rightmost[0][1] > rightmost[1][1])
    std::swap(rightmost[0], rightmost[1]);

  std::vector<std::vector<int>> rect = {leftmost[0], rightmost[0], rightmost[1],
                                        leftmost[1]};
  return rect;
}

}  // namespace Deploy
