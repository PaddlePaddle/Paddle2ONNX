//
// Created by zhengbicheng on 2022/8/11.
//

#include "PP_HumanSeg.h"

static int humanseg_target_size[2] = {192, 192};
static int humanseg_color_map[3] = {255, 0, 0};

PP_HumanSeg::PP_HumanSeg() {
    target_size = humanseg_target_size;
}

void PP_HumanSeg::predict(std::vector<float> rknn_results, cv::Mat src_img, std::string save_path) {
    cv::Mat raw_frame;
    cv::resize(src_img, raw_frame, cv::Size(target_size[0], target_size[1]), (0, 0), (0, 0), cv::INTER_LINEAR);
    std::vector<int> pred = op::argmax_by3(rknn_results, 2);
    raw_frame = display_masked_image(pred, raw_frame);
    cv::imwrite(save_path,raw_frame);
}

cv::Mat PP_HumanSeg::display_masked_image(std::vector<int> pred, cv::Mat raw_frame) {
    cv::Mat dst = raw_frame.clone();
    cv::Mat result;
    for (int row = 0; row < dst.rows; row++) {
        for (int col = 0; col < dst.cols; col++) {
            if(pred[dst.rows * row + col] == true){
                for (int i = 0; i < 3; ++i) {
                    dst.at<cv::Vec3b>(row,col)[i]=humanseg_color_map[i];
                }
            }
        }
    }

    double alpha = 0.6;
    double beta = 1 - alpha;
    double gamma = 0;
    cv::addWeighted(dst, alpha, raw_frame, beta, gamma, result, -1);
    return result;
}