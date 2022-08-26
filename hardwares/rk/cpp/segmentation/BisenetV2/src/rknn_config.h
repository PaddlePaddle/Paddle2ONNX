//
// Created by zhengbicheng on 2022/8/25.
//

#ifndef MODELTORKNN_RKNN_CONFIG_H
#define MODELTORKNN_RKNN_CONFIG_H

#include "opencv2/core/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include "rknn_api.h"
#include <sys/time.h>
#include <string.h>
#include <vector>
#include <iostream>

//std::vector<float> get_result_by_path();
std::vector<float> get_result_by_img(cv::Mat img, rknn_context ctx);

#endif //MODELTORKNN_RKNN_CONFIG_H
