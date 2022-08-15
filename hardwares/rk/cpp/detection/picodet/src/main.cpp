//
// Created by zhengbicheng on 2022/8/10.
//
#include "opencv2/core/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include "rknn_api.h"

#include "picodet.h"

#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <vector>
#include <iostream>
#include <fstream>

static inline int64_t getCurrentTimeUs() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000000 + tv.tv_usec;
}

static void dump_tensor_attr(rknn_tensor_attr *attr) {
    printf("  index=%d, name=%s, n_dims=%d, dims=[%d, %d, %d, %d], n_elems=%d, size=%d, fmt=%s, type=%s, qnt_type=%s, "
           "zp=%d, scale=%f\n",
           attr->index, attr->name, attr->n_dims, attr->dims[0], attr->dims[1], attr->dims[2], attr->dims[3],
           attr->n_elems, attr->size, get_format_string(attr->fmt), get_type_string(attr->type),
           get_qnt_type_string(attr->qnt_type), attr->zp, attr->scale);
}

char img_path[] = "./images/before/picodet_demo_input.jpg";
char model_path[] = "./model/picodet_s_320_coco_sim.rknn";
char save_path[] = "./images/after/result.jpg";
int ret = 0;
std::vector<std::string> label_names;

int main() {
    // read label_names
    printf("-> Reading label_names\n");
    std::ifstream file;
    file.open("./model/label_names.txt", std::ios_base::in);
    if (!file.is_open())
    {
        printf("Read label_names.txt failed.Exit\n");
    }
    std::string s;
    while (getline(file, s))
    {
        label_names.push_back(s);
    }
    file.close();

    // 初始化上下文，需要先创建上下文对象和读取模型文件
    printf("-> Loading model\n");
    rknn_context ctx = 0;
    ret = rknn_init(&ctx, model_path, 0, 0, NULL);
    if (ret < 0) {
        printf("rknn_init fail! ret=%d\n", ret);
        return -1;
    }

    // 获取sdk 和 驱动的版本
    printf("-> Get sdk and driver version\n");
    rknn_sdk_version sdk_ver;
    ret = rknn_query(ctx, RKNN_QUERY_SDK_VERSION, &sdk_ver, sizeof(sdk_ver));
    if (ret != RKNN_SUCC) {
        printf("rknn_query fail! ret=%d\n", ret);
        return -1;
    }
    printf("rknn_api/rknnrt version: %s, driver version: %s\n", sdk_ver.api_version, sdk_ver.drv_version);

    // 获取模型输入输出详细信息
    printf("-> Get Model Input Output Info\n");
    rknn_input_output_num io_num;
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret != RKNN_SUCC) {
        printf("rknn_query fail! ret=%d\n", ret);
        return -1;
    }
    printf("model input num: %d, output num: %d\n", io_num.n_input, io_num.n_output);

    // 获取输入参数
    printf("-> Get input tensors:\n");
    rknn_tensor_attr input_attrs[io_num.n_input];
    memset(input_attrs, 0, io_num.n_input * sizeof(rknn_tensor_attr));
    for (uint32_t i = 0; i < io_num.n_input; i++) {
        input_attrs[i].index = i;
        // query info
        ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret < 0) {
            printf("rknn_init error! ret=%d\n", ret);
            return -1;
        }
        dump_tensor_attr(&input_attrs[i]);
    }

    // 获取输出参数
    printf("-> Get output tensors:\n");
    rknn_tensor_attr output_attrs[io_num.n_output];
    memset(output_attrs, 0, io_num.n_output * sizeof(rknn_tensor_attr));
    for (uint32_t i = 0; i < io_num.n_output; i++) {
        output_attrs[i].index = i;
        // query info
        ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC) {
            printf("rknn_query fail! ret=%d\n", ret);
            return -1;
        }
        dump_tensor_attr(&output_attrs[i]);
    }

    // Get custom string
    printf("-> Get custom string\n");
    rknn_custom_string custom_string;
    ret = rknn_query(ctx, RKNN_QUERY_CUSTOM_STRING, &custom_string, sizeof(custom_string));
    if (ret != RKNN_SUCC) {
        printf("rknn_query fail! ret=%d\n", ret);
        return -1;
    }
    printf("custom string: %s\n", custom_string.string);

    unsigned char *input_data = NULL;
    rknn_tensor_type input_type = RKNN_TENSOR_UINT8;
    rknn_tensor_format input_layout = RKNN_TENSOR_NHWC;

    // Load image
    int req_height = 0;
    int req_width = 0;
    int req_channel = 0;

    switch (input_attrs[0].fmt) {
        case RKNN_TENSOR_NHWC:
            req_height = input_attrs[0].dims[1];
            req_width = input_attrs[0].dims[2];
            req_channel = input_attrs[0].dims[3];
            break;
        case RKNN_TENSOR_NCHW:
            req_height = input_attrs[0].dims[2];
            req_width = input_attrs[0].dims[3];
            req_channel = input_attrs[0].dims[1];
            break;
        default:
            printf("meet unsupported layout\n");
            return -1;
    }

    int height = 0;
    int width = 0;
    int channel = 0;

    // 使用opencv读取需要处理的图片，一定要将图片从BGR转换为RGB格式
    printf("-> Reading picture\n");
    cv::Mat src_img = imread(img_path, cv::IMREAD_COLOR);
    if (!src_img.data) {
        printf("cv::imread %s fail!\n", img_path);
        return -1;
    }

    printf("-> BGR to RGB\n");
    cv::Mat src_img_rgb;
    cv::cvtColor(src_img, src_img_rgb, cv::COLOR_BGR2RGB);

    // resize
    printf("-> Resize picture\n");
    cv::Mat input_img = src_img_rgb.clone();
    if (src_img.cols != req_width || src_img.rows != req_height) {
        printf("resize %d %d to %d %d\n", src_img.cols, src_img.rows, req_width, req_height);
        cv::resize(src_img_rgb, input_img, cv::Size(req_width, req_height), (0, 0), (0, 0), cv::INTER_LINEAR);
    }
    printf("-> Set input data\n");
    input_data = input_img.data;
    if (!input_data) {
        return -1;
    }

    // Create input tensor memory
    printf("-> Create input tensor memory");
    rknn_tensor_mem *input_mems[1];

    // default input type is int8 (normalize and quantize need compute in outside)
    // if set uint8, will fuse normalize and quantize to npu
    printf("-> Set input_attrs.type uint8");
    input_attrs[0].type = input_type;

    // default fmt is NHWC, npu only support NHWC in zero copy mode
    printf("-> Set input fun to NHWC\n");
    input_attrs[0].fmt = input_layout;
    input_mems[0] = rknn_create_mem(ctx, input_attrs[0].size_with_stride);

    // Copy input data to input tensor memory
    printf("-> Copy input data to input tensor memory");
    width = input_attrs[0].dims[2];
    int stride = input_attrs[0].w_stride;
    if (width == stride) {
        memcpy(input_mems[0]->virt_addr, input_data, width * input_attrs[0].dims[1] * input_attrs[0].dims[3]);
    } else {
        int height = input_attrs[0].dims[1];
        int channel = input_attrs[0].dims[3];
        // copy from src to dst with stride
        uint8_t *src_ptr = input_data;
        uint8_t *dst_ptr = (uint8_t *) input_mems[0]->virt_addr;
        // width-channel elements
        int src_wc_elems = width * channel;
        int dst_wc_elems = stride * channel;
        for (int h = 0; h < height; ++h) {
            memcpy(dst_ptr, src_ptr, src_wc_elems);
            src_ptr += src_wc_elems;
            dst_ptr += dst_wc_elems;
        }
    }

    // Create output tensor memory
    printf("-> Create output tensor memory\n");
    rknn_tensor_mem *output_mems[io_num.n_output];
    for (uint32_t i = 0; i < io_num.n_output; ++i) {
        // default output type is depend on model, this requires float32 to compute top5
        // allocate float32 output tensor
        int output_size = output_attrs[i].n_elems * sizeof(float);
        output_mems[i] = rknn_create_mem(ctx, output_size);
    }

    // Set input tensor memory
    printf("-> Set input tensor memory\n");
    ret = rknn_set_io_mem(ctx, input_mems[0], &input_attrs[0]);
    if (ret < 0) {
        printf("rknn_set_io_mem fail! ret=%d\n", ret);
        return -1;
    }

    // Set output tensor memory
    printf("-> Set output tensor memory\n");
    for (uint32_t i = 0; i < io_num.n_output; ++i) {
        // default output type is depend on model, this requires float32 to compute top5
        output_attrs[i].type = RKNN_TENSOR_FLOAT32;
        // set output memory and attribute
        ret = rknn_set_io_mem(ctx, output_mems[i], &output_attrs[i]);
        if (ret < 0) {
            printf("rknn_set_io_mem fail! ret=%d\n", ret);
            return -1;
        }
    }

    // Run
    printf("Begin perf ...\n");
    int64_t start_us = getCurrentTimeUs();
    ret = rknn_run(ctx, NULL);
    int64_t elapse_us = getCurrentTimeUs() - start_us;
    if (ret < 0) {
        printf("rknn run error %d\n", ret);
        return -1;
    }
    printf("Elapse Time = %.2fms, FPS = %.2f\n", elapse_us / 1000.f, 1000.f * 1000.f / elapse_us);

    // get result
    printf("-> get result\n");
    std::vector <std::vector<float>> results;
    std::vector<float> temp_result;
    for (uint32_t i = 0; i < io_num.n_output; i++) {
        temp_result.clear();
        int result_length = 1;
        for (int j = 0; j < 4; ++j) {
            result_length *= output_attrs[i].dims[j];
        }
        for (int j = 0;j < result_length; ++j) {
            temp_result.push_back(((float *) output_mems[i]->virt_addr)[j]);
        }
        results.push_back(temp_result);
    }

    printf("-> Predict\n");
    Picodet picodet = Picodet();
    picodet.predict(results, src_img, save_path,label_names);
    return 0;
}