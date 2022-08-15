# RK模型检测模型部署指南

本文档介绍在RK系列芯片上部署Picodet检测模型的步骤，具体包括：

1. 使用Paddle2ONNX将PaddleInference model转换为ONNX模型格式。
2. 运行推理脚本获得推理结果。

## 模型转换

模型版本为: [PaddleDetection 2.3分支](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.3/configs/picodet)

```text
# 进入目录
cd ./weights/onnx

# 下载picodet模型
wget https://bj.bcebos.com/paddle2onnx/model_zoo/picodet_s_320_coco.tar.gz
tar xvf picodet_s_320_coco.tar.gz
# 将Paddle模型导出为ONNX模型
paddle2onnx --model_dir ./picodet_s_320_coco \
            --model_filename model.pdmodel \
            --params_filename model.pdiparams \
            --save_file picodet_s_320_coco.onnx \
            --opset_version 12 \
            --enable_onnx_checker True \
            --input_shape_dict "{'image': [1, 3, 320, 320]}"
            
# 删除无用文件
rm -rf picodet_s_320_coco
rm -rf picodet_s_320_coco.tar.gz
```

## 运行

### 更多参数

执行以下命令

```text
python picodet_infer.py -h
```

### ONNX

推理执行以下命令

```text
python picodet_infer.py
```

### RKNN for PC

推理执行以下命令

```text
python picodet_infer.py --backend_type rk_pc
```

### RKNN for Board

推理执行以下命令

```text
sudo -E python3 picodet_infer.py --backend_type rk_board --model_path ./weights/rknn/picodet_s_320_coco_sim.rknn 
```

## 推理速度

基本信息：

- 环境：RK3588 + rknn_toolkit2 develop 最新版
- 数据集：coco2017 val数据集的前700张图片

数据表:

| 模型            | iou(0.5) | speed(ms) |
|---------------|----------|-----------|
| ONNX          | 0.451    | 41        |
| rk_board(量化前) | 0.451    | 70        |

## 查看结果

输入图片
![输入图片](./images/before/picodet_demo_input.jpg)

输出图片
![输出图片](./images/after/picodet_demo_input_rk_pc.jpg)