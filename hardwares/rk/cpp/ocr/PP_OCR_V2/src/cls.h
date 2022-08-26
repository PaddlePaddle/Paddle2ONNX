//
// Created by zhengbicheng on 2022/8/25.
//

#ifndef MODELTORKNN_CLS_H
#define MODELTORKNN_CLS_H

#include "opencv2/core/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include <algorithm>
#include <iostream>
#include "rknn_api.h"
#include "utility.h"
#include "rknn_config.h"
class Cls {
public:
    Cls(char *model_path);
    std::vector<int> cls_labels;
    std::vector<float> cls_scores;
    int predict(std::vector<cv::Mat> img_list);
private:
    rknn_context cls_ctx;
};
#endif //MODELTORKNN_CLS_H
