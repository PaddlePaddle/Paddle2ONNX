//
// Created by zhengbicheng on 2022/8/12.
//

#include "det.h"
#include "iostream"
Det::Det() {

}
Det::~Det() {

}
void Det::predict(std::vector<float> results, cv::Mat src_img, std::string save_path) {
    cv::Mat temp_src = src_img;
    boxes.clear();
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

    printf("det_db_box_thresh_ = %lf,det_db_unclip_ratio_ = %lf,det_db_score_mode_ = %lf\n",
            this->det_db_box_thresh_, this->det_db_unclip_ratio_,
            this->det_db_score_mode_);

    boxes = post_processor_.BoxesFromBitmap(
            pred_map, bit_map, this->det_db_box_thresh_, this->det_db_unclip_ratio_,
            this->det_db_score_mode_);
    boxes = post_processor_.FilterTagDetRes(boxes, 1, 1, temp_src);
    printf("boxes.size() = %lu\n",boxes.size());
//    std::vector<cv::Point> pts;
//    int count = 0;
//    for (int i = 0; i < boxes.size(); ++i) {
//        for(int j = 0;j < boxes[0].size();j++){
//            for(int k=0;k<boxes[0][0].size();k+=2){
//                pts.push_back(cv::Point(boxes[i][j][k],boxes[i][j][k+1]));
//                count++;
//            }
//            if((j+1)%2==0){
//              cv::polylines(temp_src, pts, true, cv::Scalar(255,255,0), 2,8,0);
//               pts.clear();
//            }
//        }
//    }
//
//    cv::imwrite(save_path,temp_src);
}