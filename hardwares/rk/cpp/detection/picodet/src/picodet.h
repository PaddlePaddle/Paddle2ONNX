//
// Created by zhengbicheng on 2022/8/10.
//

#ifndef PICODET_H
#define PICODET_H

#include "opencv2/core/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include "op_tools.h"
#include "image_tools.h"


class Picodet {
public:
    Picodet();

    void predict(std::vector <std::vector<float>> rknn_results, cv::Mat src_img, std::string save_path,std::vector <std::string> class_label);

private:
    int *target_size;
    int *strides;
    int nms_top_k;
    int keep_top_k;
    float score_threshold;
    float nms_threshold;
    int class_num;

    std::vector <std::vector<float>> np_score_list, np_boxes_list;
    std::vector <BboxWithID> last_result;
    void detect();
};


#endif //PICODET_H
