//
// Created by zhengbicheng on 2022/8/10.
//

#include "picodet.h"

static int picodet_target_size[2] = {320, 320};
static int picodet_strides[4] = {8, 16, 32, 64};
static int picodet_nms_top_k = 1000;
static int picodet_keep_top_k = 100;
static float picodet_score_threshold = 0.5;
static float picodet_nms_threshold = 0.5;
static int picodet_class_num = 80;

Picodet::Picodet() {
    strides = picodet_strides;
    target_size = picodet_target_size;
    nms_top_k = picodet_nms_top_k;
    keep_top_k = picodet_keep_top_k;
    score_threshold = picodet_score_threshold;
    nms_threshold = picodet_nms_threshold;
    class_num = picodet_class_num;
}

void Picodet::predict(std::vector <std::vector<float>> rknn_results, cv::Mat src_img, std::string save_path,
                      std::vector <std::string> class_label) {
    last_result.clear();
    np_score_list.clear();
    np_boxes_list.clear();
    int num_outs = (int) (rknn_results.size() / 2);
    for (int out_idx = 0; out_idx < num_outs; ++out_idx) {
        np_score_list.push_back(rknn_results[out_idx]);
        np_boxes_list.push_back(rknn_results[out_idx + num_outs]);
    }

    detect();

    printf("src_img.rows = %d,src_img.cols = %d\n",src_img.rows,src_img.cols);
    float scale_x = src_img.cols / 1.0 / target_size[1];
    float scale_y = src_img.rows / 1.0 / target_size[0];

    src_img = draw_box(src_img, last_result, class_label, scale_x, scale_y);
    cv::imwrite(save_path, src_img);
}

void Picodet::detect() {
    std::vector<float> select_scores;
    std::vector <rectangle> decode_boxes;
    for (int i = 0; i < np_score_list.size(); ++i) {
        std::vector<float> scores = np_score_list[i];
        std::vector<float> boxes = np_boxes_list[i];

        int stride = strides[i];

        float fm_h = (float) target_size[0] / stride;
        float fm_w = (float) target_size[1] / stride;

        std::vector<float> h_range, w_range;
        for (int j = 0; j < fm_h; ++j) {
            h_range.push_back((j + 0.5) * stride);
        }

        for (int j = 0; j < fm_w; ++j) {
            w_range.push_back((j + 0.5) * stride);
        }

        std::vector <rectangle> center;
        rectangle temp_rec;
        for (int j = 0; j < h_range.size(); ++j) {
            for (int k = 0; k < w_range.size(); ++k) {
                temp_rec = {h_range[k], w_range[j], h_range[k], w_range[j]};
                center.push_back(temp_rec);
            }
        }

        int reg_max = 8;
        boxes = op::softmax(boxes, reg_max);
        std::vector<int> reg_range;
        for (int j = 0; j < 8; ++j) {
            reg_range.push_back(j);
        }

        for (int j = 0; j < boxes.size(); j = j + reg_max) {
            for (int k = 0; k < reg_max; ++k) {
                boxes[j + k] = boxes[j + k] * reg_range[k];
            }
        }

        std::vector<float> box_distance;
        for (int j = 0; j < boxes.size() / reg_max; j++) {
            float sum = 0;
            for (int k = 0; k < reg_max; ++k) {
                sum += boxes[reg_max * j + k];
            }
            box_distance.push_back(sum * stride);
        }

        std::vector<float> max_scores;
        max_scores = op::get_max(scores, class_num);

        std::vector<int> topk_idx;
        topk_idx = op::argsort(max_scores, max_scores.size());

        while (topk_idx.size() > nms_top_k) {
            topk_idx.pop_back();
        }
        std::vector <rectangle> temp_center;
        std::vector<float> temp_scores;
        std::vector<float> temp_box_distance;
        for (int j = 0; j < topk_idx.size(); ++j) {
            temp_center.push_back(center[topk_idx[j]]);
            for (int k = 0; k < class_num; ++k) {
                temp_scores.push_back(scores[class_num * topk_idx[j] + k]);
            }
            for (int k = 0; k < 4; ++k) {
                temp_box_distance.push_back(box_distance[4 * topk_idx[j] + k]);
            }
        }

        std::vector <rectangle> decode_box;
        for (int j = 0; j < topk_idx.size(); ++j) {
            temp_rec = {temp_center[j].x0 - temp_box_distance[4 * j + 0],
                        temp_center[j].y0 - temp_box_distance[4 * j + 1],
                        temp_center[j].x1 + temp_box_distance[4 * j + 2],
                        temp_center[j].y1 + temp_box_distance[4 * j + 3]};
            decode_box.push_back(temp_rec);
        }

        for (int j = 0; j < temp_scores.size(); ++j) {
            select_scores.push_back(temp_scores[j]);
        }
        for (int j = 0; j < decode_box.size(); ++j) {
            decode_boxes.push_back(decode_box[j]);
        }
    }

    // nms
    for (int i = 0; i < class_num; ++i) {
        std::vector<int> mask;
        int count = 0;
        for (int j = i; j < select_scores.size(); j += class_num) {
            if (select_scores[j] > 0.5) {
                mask.push_back(j);
            }
        }
        if (mask.size() == 0)
            continue;
        std::vector <Bbox> temp_result;
        for (int j = 0; j < mask.size(); ++j) {
            Bbox t_pico_rec = {{decode_boxes[mask[j] / 80]}, select_scores[mask[j]]};
            temp_result.push_back(t_pico_rec);
        }

        temp_result = op::nms(temp_result, nms_threshold);

        for (int j = 0; j < temp_result.size(); ++j) {
            last_result.push_back({temp_result[j], {i}});
        }
    }

    printf("last_result:\n");
    for (int i = 0; i < last_result.size(); ++i) {
        printf("%f %f %f %f %d\n",last_result[i].box.rec.x0,last_result[i].box.rec.y0,
               last_result[i].box.rec.x1,last_result[i].box.rec.y1,last_result[i].id);
    }

}

