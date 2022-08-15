//
// Created by zhengbicheng on 2022/8/11.
//

#include "image_tools.h"

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
                 float scale_y) {
    for (int i = 0; i < results.size(); ++i) {
        rectangle t_rec = results[i].box.rec;
        float score = results[i].box.score;
        int id = results[i].id;
        std::string label = class_label[id];
        int x_min = results[i].box.rec.x0 * scale_x;
        int y_min = results[i].box.rec.y0 * scale_y;
        int x_max = results[i].box.rec.x1 * scale_x;
        int y_max = results[i].box.rec.y1 * scale_y;
        printf("label = %s,loc = %d %d %d %d,score = %f\n", label.data(),
               x_min, y_min, x_max, y_max, score);
        // 画框
        cv::Scalar colorRectangle(255, 0, 0);
        cv::rectangle(img, cv::Point(x_min, y_min), cv::Point(x_max, y_max), colorRectangle);

        cv::putText(img,
                    label,
                    cv::Point(x_min, y_min - 15),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.5,
                    colorRectangle,
                    1,
                    cv::LINE_AA);
        cv::putText(img,
                    std::to_string(score),
                    cv::Point(x_min, y_min),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.5,
                    colorRectangle,
                    1,
                    cv::LINE_AA);
    }
    return img;
}


//==================================================================
//函 数 名: sort_points
//功能描述: 排序点列表,默认为从小到大
//输入参数:
//      * points: 点数据集
//      * function: 按什么排序,0:按x,1:按y
//返 回 值：std::vector <cv::Point>
//作    者：郑必城
//日    期：2022-08-11
//==================================================================
bool cmpx(cv::Point const &a, cv::Point const &b) {
    return a.x < b.x;
}

bool cmpy(cv::Point const &a, cv::Point const &b) {
    return a.y < b.y;
}

void sort_points(std::vector <cv::Point> &points, int function) {
    if (function == 0) {
        std::sort(points.begin(), points.end(), cmpx);
    } else {
        std::sort(points.begin(), points.end(), cmpy);
    }
}