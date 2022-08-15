//
// Created by zhengbicheng on 2022/8/11.
//

#ifndef OP_TOOLS_H
#define OP_TOOLS_H

#include <vector>
#include <cmath>
#include <algorithm>
#include "base_config.h"
#include <iostream>

//==================================================================
//函 数 名: get_iou
//功能描述: 计算两个方框的iou
//输入参数:
//      * box1: 框1
//      * box2: 框2
//返 回 值：float
//作    者：郑必城
//日    期：2022-08-11
//==================================================================
static float get_iou(Bbox box1, Bbox box2);

namespace op {
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
    std::vector<float> softmax(std::vector<float> input, int num);

    //==================================================================
    //函 数 名: get_max
    //功能描述: 对应np.max操作,默认按行进行max
    //输入参数:
    //      * input: 需要max的数组
    //      * num: 一次需要get max读取的数据
    //返 回 值：std::vector<int>
    //作    者：郑必城
    //日    期：2022-08-11
    //==================================================================
    std::vector<float> get_max(std::vector<float> input, int num);

    //==================================================================
    //函 数 名: argsort
    //功能描述: 对应np.argsort操作,默认按行进行argsort,为了方便，这里改成从大到小放
    //输入参数:
    //      * input: 需要sort的数组
    //      * num: 一次argsort读取的数据
    //返 回 值：std::vector<int>
    //作    者：郑必城
    //日    期：2022-08-11
    //==================================================================
    std::vector<int> argsort(std::vector<float> input, int num);

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
    std::vector<int> argmax_by2(std::vector<float> input, int num);

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
    std::vector<int> argmax_by3(std::vector<float> input, int num);

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
    std::vector <Bbox> nms(std::vector <Bbox> &boxes, float threshold);


}
#endif //OP_TOOLS_H
