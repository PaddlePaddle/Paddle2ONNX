#include "op_tools.h"

//==================================================================
//函 数 名: softmax
//功能描述: 对应softmax算子,默认按行进行softmax
//输入参数:
//      * input: 需要softmax的数组
//      * num: 一次softmax读取的数据
//返 回 值: std::vector<float>
//作    者: 郑必城
//日    期: 2022-08-11
//==================================================================
std::vector<float> op::softmax(std::vector<float> input, int num) {
    std::vector<float> result;
    int batch = input.size() / num;
    for (int i = 0; i < batch; ++i) {
        float MAX = input[i * num];
        float total = 0;
        for (int j = i * num; j < (i + 1) * num; ++j) {
            MAX = std::max(input[j], MAX);
        }
        for (int j = i * num; j < (i + 1) * num; ++j) {
            total += exp(input[j] - MAX);
        }
        for (int j = i * num; j < (i + 1) * num; ++j) {
            result.push_back(exp(input[j] - MAX) / total);
        }
    }
    return result;
}

//==================================================================
//函 数 名: argsort
//功能描述: 对应np.argsort操作,默认按行进行argsort,为了方便，这里改成从大到小放
//
//输入参数:
//      * input: 需要sort的数组
//      * num: 一次argsort读取的数据
//返 回 值：std::vector<int>
//作    者：郑必城
//日    期：2022-08-11
//==================================================================
std::vector<int> op::argsort(std::vector<float> input, int num) {
    const int array_len(input.size());
    std::vector<int> array_index(array_len, 0);
    for (int i = 0; i < array_len; i = i + num) {
        for (int j = 0; j < num; ++j) {
            array_index[i + j] = j;
        }
        std::sort(array_index.begin() + i, array_index.begin() + i + num,
                  [&input](int pos1, int pos2) { return (input[pos1] > input[pos2]); });
    }
    return array_index;
}

//==================================================================
//函 数 名: get_max
//功能描述: 对应np.max操作,默认按行进行max
//
//输入参数:
//      * input: 需要max的数组
//      * num: 一次需要get max读取的数据
//返 回 值：std::vector<int>
//作    者：郑必城
//日    期：2022-08-11
//==================================================================
std::vector<float> op::get_max(std::vector<float> input, int num) {
    std::vector<float> result;
    int batch = input.size() / num;
    for (int i = 0; i < batch; ++i) {
        float MAX = input[i * num];
        for (int j = i * num; j < (i + 1) * num; ++j) {
            MAX = std::max(input[j], MAX);
        }
        result.push_back(MAX);
    }
    return result;
}

//==================================================================
//函 数 名: get_iou
//功能描述: 计算两个方框的iou
//
//输入参数:
//      * box1: 框1
//      * box2: 框2
//返 回 值：float
//作    者：郑必城
//日    期：2022-08-11
//==================================================================
static float get_iou(Bbox box1, Bbox box2) {
    float x1 = std::max(box1.rec.x0, box2.rec.x0);
    float y1 = std::max(box1.rec.y0, box2.rec.y0);
    float x2 = std::min(box1.rec.x1, box2.rec.x1);
    float y2 = std::min(box1.rec.y1, box2.rec.y1);
    float w = std::max((float) 0, x2 - x1);
    float h = std::max((float) 0, y2 - y1);
    float over_area = w * h;
    return over_area / ((box1.rec.x1 - box1.rec.x0) * (box1.rec.y1 - box1.rec.y0) +
                        (box2.rec.x1 - box2.rec.x0) * (box2.rec.y1 - box2.rec.y0) - over_area);
}

//==================================================================
//函 数 名: nms
//功能描述: 对应nms操作
//输入参数:
//      * boxes: 需要nms的方框数据
//      * threshold: 置信度
//返 回 值：std::vector<Bbox>
//作    者：郑必城
//日    期：2022-08-11
//==================================================================
static bool sort_score(Bbox box1, Bbox box2) {
    return box1.score > box2.score ? true : false;
}

std::vector <Bbox> op::nms(std::vector <Bbox> &boxes, float threshold) {
    std::vector <Bbox> resluts;
    std::sort(boxes.begin(), boxes.end(), sort_score);
    while (boxes.size() > 0) {
        resluts.push_back(boxes[0]);
        int index = 1;
        while (index < boxes.size()) {
            float iou_value = get_iou(boxes[0], boxes[index]);
            if (iou_value > threshold) {
                boxes.erase(boxes.begin() + index);
            } else {
                index++;
            }
        }
        boxes.erase(boxes.begin());
    }
    return resluts;
}

//==================================================================
//函 数 名: argmax_by2
//功能描述: 对应np.argmax 二维行操作,默认按行进行argmax
//输入参数:
//      * input: 需要sort的数组
//      * num: 一次argsort读取的数据
//返 回 值：std::vector<int>
//作    者：郑必城
//日    期：2022-08-11
//==================================================================
std::vector<int> op::argmax_by2(std::vector<float> input, int num) {
    std::vector<int> result;
    for (int i = 0; i < input.size(); i += num) {
        int max = i;
        for (int j = i; j < i + num; ++j) {
            if (input[i] > input[max]) {
                max = i;
            }
        }
        result.push_back(max);
    }
    return result;
}

//==================================================================
//函 数 名: argmax_by3
//功能描述: 对应np.argmax 三维操作,默认按行进行argmax
//输入参数:
//      * input: 需要sort的数组
//      * num: 一次argsort读取的数据
//返 回 值：std::vector<int>
//作    者：郑必城
//日    期：2022-08-11
//==================================================================
std::vector<int> op::argmax_by3(std::vector<float> input, int num) {
    std::vector<int> result;
    int length = input.size() / num;
    for (int i = 0; i < length; ++i) {
        int max = 0;
        for (int j = 0; j < num; ++j) {
            if (input[max * length + i] <= input[j * length + i]){
                max = j;
            }
        }
        result.push_back(max);
    }
    return result;
}

