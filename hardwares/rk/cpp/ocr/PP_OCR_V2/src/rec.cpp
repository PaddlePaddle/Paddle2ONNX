//
// Created by zhengbicheng on 2022/8/26.
//

#include "rec.h"

using namespace PaddleOCR;

void Rec::load_label_list(std::string label_file) {
    label_list_ = Utility::ReadDict(label_file);
    label_list_.insert(label_list_.begin(), " ");
    label_list_.push_back(" ");
}

Rec::Rec(char *model_path) {
    // 初始化上下文，需要先创建上下文对象和读取模型文件
    printf("-> Loading model\n");
    rec_ctx = 0;
    int ret = rknn_init(&rec_ctx, model_path, 0, 0, NULL);
    if (ret < 0) {
        printf("rknn_init fail! ret=%d\n", ret);
        exit(-1);
    }
}

void Rec::predict(std::vector <cv::Mat> img_list) {
    rec_texts.clear();
    rec_text_scores.clear();

    rec_texts.resize(img_list.size(), "");
    rec_text_scores.resize(img_list.size(), 0);

    std::vector<int> rec_image_shape_ = {3, 32, 960};
    int img_num = img_list.size();
    std::vector<float> width_list;
    for (int i = 0; i < img_num; i++) {
        width_list.push_back(float(img_list[i].cols) / img_list[i].rows);
    }
    std::vector<int> indices = Utility::argsort(width_list);
    std::vector <cv::Mat> norm_img_batch;

    for (int i = 0; i < img_num; i++) {
        int imgH = 32;
        int imgW = 960;
        float max_wh_ratio = imgW * 1.0 / imgH;
        int h = img_list[indices[i]].rows;
        int w = img_list[indices[i]].cols;
        float wh_ratio = w * 1.0 / h;
        max_wh_ratio = max(max_wh_ratio, wh_ratio);

        int batch_width = imgW;

        cv::Mat srcimg;
        img_list[indices[i]].copyTo(srcimg);
        cv::Mat resize_img;
        this->resize_op_.Run(srcimg, resize_img, max_wh_ratio, false, rec_image_shape_);
        norm_img_batch.push_back(resize_img);
    }

//    for (int i = 0; i < norm_img_batch.size(); ++i) {
//        cv::imwrite("./images/after/" + std::to_string(i)+".jpg",norm_img_batch[i]);
//    }

    for (int i = 0; i < img_num; ++i) {
        std::vector<int> predict_shape = {1, 240, 6625};
        std::vector<float> results = get_result_by_img(norm_img_batch[indices[i]], rec_ctx);

        for (int m = 0; m < predict_shape[0]; ++m) {
            std::string str_res;
            int argmax_idx;
            int last_index = 0;
            float score = 0.f;
            int count = 0;
            float max_value = 0.0f;

            for (int n = 0; n < predict_shape[1]; ++n) {
                // get idx
                argmax_idx = int(Utility::argmax(
                        &results[(m * predict_shape[1] + n) * predict_shape[2]],
                        &results[(m * predict_shape[1] + n + 1) * predict_shape[2]]));
                // get score
                max_value = float(*std::max_element(
                        &results[(m * predict_shape[1] + n) * predict_shape[2]],
                        &results[(m * predict_shape[1] + n + 1) * predict_shape[2]]));
                if (argmax_idx > 0 && (!(n > 0 && argmax_idx == last_index))) {
                    score += max_value;
                    count += 1;
                    str_res += label_list_[argmax_idx];
                }
                last_index = argmax_idx;
            }
            score /= count;
            if (std::isnan(score)) {
                continue;
            }
            rec_texts[indices[i + m]] = str_res;
            rec_text_scores[indices[i + m]] = score;
        }
    }
}