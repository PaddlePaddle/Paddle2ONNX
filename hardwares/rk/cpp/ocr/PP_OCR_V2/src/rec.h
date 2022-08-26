//
// Created by zhengbicheng on 2022/8/26.
//

#ifndef MODELTORKNN_REC_H
#define MODELTORKNN_REC_H

#include "utility.h"
#include "rknn_config.h"
#include "rknn_api.h"
#include "preprocess_op.h"
class Rec {
public:
    Rec(char *model_path);
    void predict(std::vector<cv::Mat> img_list);
    void load_label_list(std::string label_file);
    std::vector<std::string> rec_texts;
    std::vector<float> rec_text_scores;
private:
    rknn_context rec_ctx;
    std::vector<std::string> label_list_;
    PaddleOCR::CrnnResizeImg resize_op_;
};


#endif //MODELTORKNN_REC_H
