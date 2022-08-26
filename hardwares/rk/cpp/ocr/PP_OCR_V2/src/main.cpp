//
// Created by zhengbicheng on 2022/8/10.
//
#include "opencv2/core/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

#include "rec.h"
#include "det.h"
#include "cls.h"

#include "utility.h"
#include "rknn_config.h"
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>

using namespace PaddleOCR;

char img_path[] = "./images/before/lite_demo_input.png";
char det_model_path[] = "./model/PP_OCR_v2_det.rknn";
char cls_model_path[] = "./model/PP_OCR_v2_cls.rknn";
char rec_model_path[] = "./model/PP_OCR_v2_rec.rknn";
char label_path[] = "./model/ppocr_keys_v1.txt";
std::string save_path = "./images/after/result.jpg";
std::vector<std::string> label_names;


int main() {
    int ret;

    // 初始化上下文，需要先创建上下文对象和读取模型文件
    printf("-> Loading model\n");
    rknn_context det_ctx = 0;
    ret = rknn_init(&det_ctx, det_model_path, 0, 0, NULL);
    if (ret < 0) {
        printf("rknn_init fail! ret=%d\n", ret);
        return -1;
    }

    cv::Mat src_img = cv::imread(img_path);
    cv::resize(src_img, src_img, cv::Size(960, 960), (0, 0), (0, 0), cv::INTER_LINEAR);
    std::vector<float> results = get_result_by_img(src_img,det_ctx);


    printf("-> PP_OCRV2_det\n");
    std::vector<cv::Mat> img_list;
    Det *det = new Det();
    det->predict(results,src_img,save_path);
    for (int j = 0; j < det->boxes.size(); j++) {
        cv::Mat crop_img;
        crop_img = Utility::GetRotateCropImage(src_img, det->boxes[j]);
        img_list.push_back(crop_img);
        cv::imwrite("./images/after/" + std::to_string(j)+".jpg",crop_img);
    }
    delete det;



    Cls *cls = new Cls(cls_model_path);
    cls->predict(img_list);
    for (int i = 0; i < img_list.size(); i++) {
          if (cls->cls_labels[i] % 2 == 1 && cls->cls_scores[i] > 0.5) {
            cv::rotate(img_list[i], img_list[i], 1);
          }
    }
    delete cls;


    Rec *rec = new Rec(rec_model_path);
    rec->load_label_list(label_path);
    rec->predict(img_list);

    for (int i = 0; i < rec->rec_text_scores.size(); ++i) {
        if (rec->rec_text_scores[i] > 0.85){
            printf("%s: %f\n",rec->rec_texts[i].data(),rec->rec_text_scores[i]);
        }
    }

    return 0;
}