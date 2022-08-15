//
// Created by zhengbicheng on 2022/8/11.
//

#ifndef PP_HUMANSEG_H
#define PP_HUMANSEG_H

#include "opencv2/core/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include "op_tools.h"
#include "image_tools.h"

class PP_HumanSeg {
public:
    PP_HumanSeg();

    void predict(std::vector<float> rknn_results, cv::Mat src_img, std::string save_path);

private:
    int *target_size;

    cv::Mat display_masked_image(std::vector<int> pred, cv::Mat raw_frame);
};


#endif //PP_HUMANSEG_H
