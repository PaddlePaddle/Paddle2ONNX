//
// Created by zhengbicheng on 2022/8/10.
//
#include "opencv2/core/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include "rknn_config.h"
#include "utility.h"
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>

char img_path[] = "./images/before/bisenet_demo_input.jpeg";
char model_path[] = "./weights/bisenet.rknn";

std::vector<int> get_color_map_list(int num_classes) {
    int base_color = 254 / num_classes;
    std::vector<int> base;
    for (int i = 0; i < num_classes; ++i) {
        base.push_back(i * base_color);
    }
    return base;
}

int main() {
    // 初始化上下文，需要先创建上下文对象和读取模型文件
    printf("-> Loading model\n");
    rknn_context ctx = 0;
    int ret = rknn_init(&ctx, model_path, 0, 0, NULL);
    if (ret < 0) {
        printf("rknn_init fail! ret=%d\n", ret);
        return -1;
    }

    cv::Mat src_img = cv::imread(img_path);
    cv::resize(src_img, src_img, cv::Size(1024, 1024), (0, 0), (0, 0), cv::INTER_LINEAR);

    std::vector<float> results_1 = get_result_by_img(src_img, ctx);
    for (int i = 0; i < 10; ++i) {
        printf("%f ", results_1[i]);
    }
    printf("\n");

    std::vector<float> results_2;
    for (int i = 0; i < 1024; ++i) {
        for (int j = 0; j < 1024; ++j) {
            for (int k = 0; k < 19; ++k) {
                results_2.push_back(results_1[1024 * 1024 * k + j + 1024 * i]);
            }
        }
    }

    std::vector<int> results_3;
    for (int i = 0; i < 1024 * 1024; ++i) {
        results_3.push_back(PaddleOCR::Utility::argmax(results_2.begin() + 19 * i, results_2.begin() + 19 * (i + 1)));
    }

    for (int i = 0; i < 10; ++i) {
        printf("%d ", results_3[i]);
    }
    printf("\n");

    std::vector<int> base_color = get_color_map_list(19);

    cv::Mat dst = src_img.clone();
    for (int row = 0; row < dst.rows; row++) {
        for (int col = 0; col < dst.cols; col++) {
            int color = base_color[results_3[row * 1024 + col]];
            for (int i = 0; i < 3; ++i) {
                dst.at<cv::Vec3b>(row, col)[i] = color;
            }
        }
    }

    cv::imwrite("./images/after/results.jpg",dst);

    return 0;
}