# RK模型检测模型部署指南
本文档介绍在RK系列芯片上部署Picodet检测模型的步骤，具体包括：
1. 使用Paddle2ONNX将PaddleInference model转换为ONNX模型格式。 
2. 运行推理脚本获得推理结果。

## 模型转换
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
sudo -E python3 picodet_infer.py --backend_type rk_board --model_path ./weights/rknn/picodet_s_320_coco.rknn 
```