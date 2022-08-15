//
// Created by zhengbicheng on 2022/8/11.
//

#ifndef IMAGE_TOOLS_H
#define IMAGE_TOOLS_H

#include "base_config.h"
#include "opencv2/core/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <algorithm>
#include <iostream>
//==================================================================
//函 数 名: draw_box
//功能描述: 根据box在图片上画画
//输入参数:
//      * img: 图片数据
//      * results: 方框数据
//      * class_label: 标签数据
//      * scale_x: 缩放比例
//      * scale_y: 缩放比例
//返 回 值：cv::Mat
//作    者：郑必城
//日    期：2022-08-11
//==================================================================
cv::Mat draw_box(cv::Mat img, std::vector <BboxWithID> results, std::vector <std::string> class_label, float scale_x,
                 float scale_y);

void sort_points(std::vector <cv::Point> &points, int function);

#endif //IMAGE_TOOLS_H
