//
// Created by zhengbicheng on 2022/8/12.
//

#ifndef DET_H
#define DET_H

#include "opencv2/core/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include <algorithm>
#include <iostream>

#include "postprocess_op.h"

class Det {
public:
    Det();
    ~Det();
    void predict(std::vector<float> results, cv::Mat src_img, std::string save_path);
    std::vector<std::vector<std::vector<int>>> boxes;

private:
    std::vector<int> target_size = {960, 960};

    double det_db_thresh_ = 0.3;
    double det_db_box_thresh_ = 0.5;
    double det_db_unclip_ratio_ = 2.0;
    std::string det_db_score_mode_ = "slow";
    bool use_dilation_ = false;

    PaddleOCR::PostProcessor post_processor_;
};


#endif //DET_H
