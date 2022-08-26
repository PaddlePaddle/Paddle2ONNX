//
// Created by zhengbicheng on 2022/8/25.
//

#include "cls.h"

Cls::Cls(char *model_path) {
    // 初始化上下文，需要先创建上下文对象和读取模型文件
    printf("-> Loading model\n");
    cls_ctx = 0;
    int ret = rknn_init(&cls_ctx, model_path, 0, 0, NULL);
    if (ret < 0) {
        printf("rknn_init fail! ret=%d\n", ret);
        exit(-1);
    }
}

int Cls::predict(std::vector <cv::Mat> img_list) {
    int ret;
    cls_labels.clear();
    cls_scores.clear();
    int img_num = img_list.size();
    std::vector<int> cls_image_shape = {3, 32, 960};
    for(int i=0;i<img_num;i++){
        std::vector<float> results = get_result_by_img(img_list[i],cls_ctx);
        int label = int(PaddleOCR::Utility::argmax(&results[0], &results[1]));
        float score = float(*std::max_element(&results[0],&results[1]));
        cls_labels.push_back(label);
        cls_scores.push_back(score);
    }
    return 0;
}