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
  if (model_arch_ == "DET") {
    det_db_thresh_ = parser.Get<double>("det_db_thresh");
    det_db_box_thresh_ = parser.Get<double>("det_db_box_thresh");
    det_db_unclip_ratio_ = parser.Get<double>("det_db_unclip_ratio");
  }
  if (model_arch_ == "CRNN") {
    std::string path = parser.Get<std::string>("path");
    ReadDict(path);
    label_list_.insert(label_list_.begin(), "#");
    label_list_.push_back(" ");
  }
}

bool PaddleOcrPostProc::Run(const std::vector<DataBlob> &outputs,
                            const std::vector<ShapeInfo> &shape_infos,
                            std::vector<PaddleOcrResult> *ocr_results) {
  ocr_results->clear();
  if (model_arch_ == "DET") {
    DetPostProc(outputs, shape_infos, ocr_results);
    return true;
  }
  if (model_arch_ == "CLS") {
    ClsPostProc(outputs, ocr_results);
    return true;
  }
  if (model_arch_ == "CRNN") {
    CrnnPostProc(outputs, ocr_results);
  }
}

bool PaddleOcrPostProc::GetRotateCropImage(
      const cv::Mat &srcimage,
      const std::vector<std::vector<std::vector<int>>> &boxes,
      std::vector<cv::Mat> *imgs) {
  int img_num = boxes.size();
  imgs->clear();
  for (int i = 0; i < img_num; i++) {
    std::vector<std::vector<int>> box;
    box = boxes[i];
    cv::Mat image;
    srcimage.copyTo(image);
    std::vector<std::vector<int>> points = box;
    int x_collect[4] = {box[0][0], box[1][0], box[2][0], box[3][0]};
    int y_collect[4] = {box[0][1], box[1][1], box[2][1], box[3][1]};
    int left = static_cast<int>(*std::min_element(x_collect, x_collect + 4));
    int right = static_cast<int>(*std::max_element(x_collect, x_collect + 4));
    int top = static_cast<int>(*std::min_element(y_collect, y_collect + 4));
    int bottom = static_cast<int>(*std::max_element(y_collect, y_collect + 4));

    cv::Mat img_crop;
    image(cv::Rect(left, top, right - left, bottom - top)).copyTo(img_crop);

    for (int i = 0; i < points.size(); i++) {
      points[i][0] -= left;
      points[i][1] -= top;
    }

    int img_crop_width = static_cast<int>(sqrt(pow(points[0][0] -
                    points[1][0], 2) + pow(points[0][1] - points[1][1], 2)));
    int img_crop_height = static_cast<int>(sqrt(pow(points[0][0] -
                    points[3][0], 2) + pow(points[0][1] - points[3][1], 2)));

    cv::Point2f pts_std[4];
    pts_std[0] = cv::Point2f(0., 0.);
    pts_std[1] = cv::Point2f(img_crop_width, 0.);
    pts_std[2] = cv::Point2f(img_crop_width, img_crop_height);
    pts_std[3] = cv::Point2f(0.f, img_crop_height);

    cv::Point2f pointsf[4];
    pointsf[0] = cv::Point2f(points[0][0], points[0][1]);
    pointsf[1] = cv::Point2f(points[1][0], points[1][1]);
    pointsf[2] = cv::Point2f(points[2][0], points[2][1]);
    pointsf[3] = cv::Point2f(points[3][0], points[3][1]);

    cv::Mat M = cv::getPerspectiveTransform(pointsf, pts_std);

    cv::Mat dst_img;
    cv::warpPerspective(img_crop, dst_img, M,
                        cv::Size(img_crop_width, img_crop_height),
                        cv::BORDER_REPLICATE);
    if (static_cast<float>(dst_img.rows) >=
        static_cast<float>(dst_img.cols) * 1.5) {
      cv::Mat srcCopy = cv::Mat(dst_img.rows, dst_img.cols, dst_img.depth());
      cv::transpose(dst_img, srcCopy);
      cv::flip(srcCopy, srcCopy, 0);
      imgs->insert(imgs->begin(), srcCopy);
    } else {
      imgs->insert(imgs->begin(), dst_img);
    }
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
      cbuf[j] = (unsigned char)((output_data[i * n + j]) * 255);
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
                    det_db_box_thresh_,
                    det_db_unclip_ratio_,
                    &ocr_result);
    FilterTagDetRes(shape_info, &ocr_result);
    ocr_results->push_back(ocr_result);
    return true;
  }
}

bool PaddleOcrPostProc::ClsPostProc(const std::vector<DataBlob> &outputs,
                            std::vector<PaddleOcrResult> *ocr_results) {
  DataBlob output_blob = outputs[0];
  float *output_data = reinterpret_cast<float*>(output_blob.data.data());
  std::vector<int> out_shape = output_blob.shape;
  int batch_size = out_shape[0];
  int output_num = 1;
  for (int i = 1; i < out_shape.size(); i++) {
    output_num *= out_shape[i];
  }
  for (int i = 0; i < batch_size; i++) {
    float cls_score = 0;
    int label = 0;
    PaddleOcrResult ocr_result;
    for (int j = 0; j < output_num; j++) {
      if (output_data[j + i * output_num] > cls_score) {
        cls_score = output_data[j + i * output_num];
        label = j;
      }
    }
    ocr_result.cls_score = cls_score;
    ocr_result.label = label;
    ocr_results->push_back(ocr_result);
  }
}

bool PaddleOcrPostProc::CrnnPostProc(const std::vector<DataBlob> &outputs,
                            std::vector<PaddleOcrResult> *ocr_results) {
  ocr_results->clear();
  DataBlob output_blob = outputs[0];
  float *output_data = reinterpret_cast<float*>(output_blob.data.data());
  std::vector<int> output_shape = output_blob.shape;
  for (int i = 0; i < output_shape[0]; i++) {
    PaddleOcrResult ocr_result;
    int argmax_idx;
    int size = output_shape[1] * output_shape[2];
    int last_index = 0;
    float crnn_score = 0.f;
    int count = 0;
    float max_value = 0.0f;
    for (int j = 0; j < output_shape[1]; j++) {
      int fisrt = i * size + j * output_shape[2];
      int last = i * size + (j + 1) * output_shape[2];
      argmax_idx =
        std::distance(output_data + fisrt,
        std::max_element(output_data + fisrt, output_data + last));
      max_value = static_cast<float>(*std::max_element(output_data + fisrt,
        output_data + last));
      if (argmax_idx > 0 && (!(i > 0 && argmax_idx == last_index))) {
        crnn_score += max_value;
        count += 1;
        ocr_result.str_res.push_back(label_list_[argmax_idx]);
      }
      last_index = argmax_idx;
    }
    crnn_score /= count;
    ocr_result.crnn_score;
    ocr_results->push_back(ocr_result);
  }
}

bool PaddleOcrPostProc::ReadDict(const std::string &path) {
  std::ifstream in(path);
  std::string line;
  if (in) {
    while (getline(in, line)) {
      label_list_.push_back(line);
    }
  } else {
    std::cout << "no such label file: " << path << ", exit the program..."
              << std::endl;
    return false;
  }
  return true;
}

bool PaddleOcrPostProc::BoxesFromBitmap(
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
    auto array = GetMiniBoxes(box, &ssid);
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
    auto cliparray = GetMiniBoxes(clipbox, &ssid);
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
  return true;
}

bool PaddleOcrPostProc::FilterTagDetRes(const ShapeInfo &shape_info,
                                  PaddleOcrResult *ocr_result) {
  int ori_w = shape_info.shape[0][0];
  int ori_h = shape_info.shape[0][1];
  int resize_w = shape_info.shape[1][0];
  int resize_h = shape_info.shape[1][1];
  float ratio_w = (float)resize_w / (float)ori_w;  // NOLINT
  float ratio_h = (float)resize_h / (float)(ori_h);  // NOLINT

  std::vector<std::vector<std::vector<int>>> root_points;
  std::vector<std::vector<std::vector<int>>> boxes;
  boxes = ocr_result->boxes;
  ocr_result->boxes.clear();
  for (int n = 0; n < boxes.size(); n++) {
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
  ocr_result->boxes = root_points;
  return true;
}

std::vector<std::vector<float>> PaddleOcrPostProc::Mat2Vector(cv::Mat mat) {
  std::vector<std::vector<float>> img_vec;
  std::vector<float> tmp;

  for (int i = 0; i < mat.rows; ++i) {
    tmp.clear();
    for (int j = 0; j < mat.cols; ++j) {
      tmp.push_back(mat.at<float>(i, j));
    }
    img_vec.push_back(tmp);
  }
  return img_vec;
}

bool PaddleOcrPostProc::XsortFp32(std::vector<float> a, std::vector<float> b) {
  if (a[0] != b[0])
    return a[0] < b[0];
  return false;
}

bool PaddleOcrPostProc::XsortInt(std::vector<int> a, std::vector<int> b) {
  if (a[0] != b[0])
    return a[0] < b[0];
  return false;
}

bool PaddleOcrPostProc::GetContourArea(
                                  const std::vector<std::vector<float>> &box,
                                  float unclip_ratio, float *distance) {
  int pts_num = 4;
  float area = 0.0f;
  float dist = 0.0f;
  for (int i = 0; i < pts_num; i++) {
    area += box[i][0] * box[(i + 1) % pts_num][1] -
            box[i][1] * box[(i + 1) % pts_num][0];
    dist += sqrtf((box[i][0] - box[(i + 1) % pts_num][0]) *
                      (box[i][0] - box[(i + 1) % pts_num][0]) +
                  (box[i][1] - box[(i + 1) % pts_num][1]) *
                      (box[i][1] - box[(i + 1) % pts_num][1]));
  }
  area = fabs(static_cast<float>(area / 2.0));

  *distance = area * unclip_ratio / dist;
  return true;
}

cv::RotatedRect PaddleOcrPostProc::UnClip(std::vector<std::vector<float>> box,
                                      const float &unclip_ratio) {
  float distance = 1.0;

  GetContourArea(box, unclip_ratio, &distance);

  ClipperLib::ClipperOffset offset;
  ClipperLib::Path p;
  p << ClipperLib::IntPoint((int)(box[0][0]), int(box[0][1]))  // NOLINT
    << ClipperLib::IntPoint((int)(box[1][0]), int(box[1][1]))  // NOLINT
    << ClipperLib::IntPoint((int)(box[2][0]), int(box[2][1]))  // NOLINT
    << ClipperLib::IntPoint((int)(box[3][0]), int(box[3][1]));  // NOLINT
  offset.AddPath(p, ClipperLib::jtRound, ClipperLib::etClosedPolygon);

  ClipperLib::Paths soln;
  offset.Execute(soln, distance);
  std::vector<cv::Point2f> points;

  for (int j = 0; j < soln.size(); j++) {
    for (int i = 0; i < soln[soln.size() - 1].size(); i++) {
      points.emplace_back(soln[j][i].X, soln[j][i].Y);
    }
  }
  cv::RotatedRect res;
  if (points.size() <= 0) {
    res = cv::RotatedRect(cv::Point2f(0, 0), cv::Size2f(1, 1), 0);
  } else {
    res = cv::minAreaRect(points);
  }
  return res;
}

float PaddleOcrPostProc::BoxScoreFast(
                                  std::vector<std::vector<float>> box_array,
                                  cv::Mat pred) {
  auto array = box_array;
  int width = pred.cols;
  int height = pred.rows;

  float box_x[4] = {array[0][0], array[1][0], array[2][0], array[3][0]};
  float box_y[4] = {array[0][1], array[1][1], array[2][1], array[3][1]};

  int xmin = clamp(static_cast<int>(
    std::floor(*(std::min_element(box_x, box_x + 4)))), 0, width - 1);
  int xmax = clamp(static_cast<int>(
    std::ceil(*(std::max_element(box_x, box_x + 4)))), 0, width - 1);
  int ymin = clamp(static_cast<int>(
    std::floor(*(std::min_element(box_y, box_y + 4)))), 0, height - 1);
  int ymax = clamp(static_cast<int>(
    std::ceil(*(std::max_element(box_y, box_y + 4)))), 0, height - 1);

  cv::Mat mask;
  mask = cv::Mat::zeros(ymax - ymin + 1, xmax - xmin + 1, CV_8UC1);

  cv::Point root_point[4];
  root_point[0] = cv::Point(static_cast<int>(array[0][0]) - xmin,
                            static_cast<int>(array[0][1]) - ymin);
  root_point[1] = cv::Point(static_cast<int>(array[1][0]) - xmin,
                            static_cast<int>(array[1][1]) - ymin);
  root_point[2] = cv::Point(static_cast<int>(array[2][0]) - xmin,
                            static_cast<int>(array[2][1]) - ymin);
  root_point[3] = cv::Point(static_cast<int>(array[3][0]) - xmin,
                            static_cast<int>(array[3][1]) - ymin);
  const cv::Point *ppt[1] = {root_point};
  int npt[] = {4};
  cv::fillPoly(mask, ppt, npt, 1, cv::Scalar(1));
  cv::Mat croppedImg;
  pred(cv::Rect(xmin, ymin, xmax - xmin + 1, ymax - ymin + 1))
      .copyTo(croppedImg);
  auto score = cv::mean(croppedImg, mask)[0];
  return score;
}

std::vector<std::vector<float>>
PaddleOcrPostProc::GetMiniBoxes(cv::RotatedRect box, float *ssid) {
  *ssid = std::max(box.size.width, box.size.height);
  cv::Mat points;
  cv::boxPoints(box, points);
  auto array = Mat2Vector(points);
  std::sort(array.begin(), array.end(), XsortFp32);
  std::vector<float> idx1 = array[0], idx2 = array[1], idx3 = array[2],
                     idx4 = array[3];
  if (array[3][1] <= array[2][1]) {
    idx2 = array[3];
    idx3 = array[2];
  } else {
    idx2 = array[2];
    idx3 = array[3];
  }
  if (array[1][1] <= array[0][1]) {
    idx1 = array[1];
    idx4 = array[0];
  } else {
    idx1 = array[0];
    idx4 = array[1];
  }

  array[0] = idx1;
  array[1] = idx2;
  array[2] = idx3;
  array[3] = idx4;
  return array;
}

std::vector<std::vector<int>>
PaddleOcrPostProc::OrderPointsClockwise(std::vector<std::vector<int>> pts) {
  std::vector<std::vector<int>> box = pts;
  std::sort(box.begin(), box.end(), XsortInt);

  std::vector<std::vector<int>> leftmost = {box[0], box[1]};
  std::vector<std::vector<int>> rightmost = {box[2], box[3]};

  if (leftmost[0][1] > leftmost[1][1])
    std::swap(leftmost[0], leftmost[1]);

  if (rightmost[0][1] > rightmost[1][1])
    std::swap(rightmost[0], rightmost[1]);

  std::vector<std::vector<int>> rect = {leftmost[0], rightmost[0],
                                        rightmost[1], leftmost[1]};
  return rect;
}

}  // namespace Deploy
