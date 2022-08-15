//
// Created by zhengbicheng on 2022/8/12.
//

#include "det.h"

Det::Det() {

}

void Det::predict(std::vector<float> results, cv::Mat src_img, std::string save_path) {
    int n = results.size();
    std::vector<float> pred(n, 0.0);
    std::vector<unsigned char> cbuf(n, ' ');
    for (int i = 0; i < n; i++) {
        pred[i] = float(results[i]);
        cbuf[i] = (unsigned char) ((results[i]) * 255);
    }

    cv::Mat cbuf_map(target_size[0], target_size[1], CV_8UC1, (unsigned char *) cbuf.data());
    cv::Mat pred_map(target_size[0], target_size[1], CV_32F, (float *) pred.data());

    const double threshold = det_db_thresh_ * 255;
    const double maxvalue = 255;

    cv::Mat bit_map;
    cv::threshold(cbuf_map, bit_map, threshold, maxvalue, cv::THRESH_BINARY);
    if (use_dilation_) {
        cv::Mat dila_ele =
                cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2));
        cv::dilate(bit_map, bit_map, dila_ele);
    }

    std::vector < std::vector < std::vector < int>>> boxes;
    boxes = post_processor_.BoxesFromBitmap(
            pred_map, bit_map, this->det_db_box_thresh_, this->det_db_unclip_ratio_,
            this->det_db_score_mode_);
    boxes = post_processor_.FilterTagDetRes(boxes, 1, 1, src_img);

    for (int i = 0; i < boxes.size(); ++i) {
        cv::polylines(src_img, boxes[i], true, cv::Scalar(0,0,0), 2);
    }

    cv::imwrite(save_path,src_img);
}