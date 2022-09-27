# RK模型分割模型部署指南
本文档介绍在RK系列芯片上部署Bisenet模型的步骤，具体包括：
1. 使用Paddle2ONNX将PaddleInference model转换为ONNX模型格式。 
2. 运行推理脚本获得推理结果。

## 模型转换
```text
# 下载ONNX模型
# 进入目录
cd ./weights
mkdir onnx
cd onnx
wget https://paddlelite-demo.bj.bcebos.com/onnx%26rknn2_model/bisenet.onnx

# 下载RKNN模型
# 进入目录
cd ./weights
mkdir rknn
cd rknn
wget https://paddlelite-demo.bj.bcebos.com/onnx%26rknn2_model/bisenet.rknn
```

## 运行
### 更多参数
执行以下命令
```text
python bisenet_infer.py -h
```
### ONNX
推理执行以下命令
```text
python bisenet_infer.py
```

### RKNN for PC
推理执行以下命令
```text
python bisenet_infer.py --backend_type rk_pc
```
### RKNN for Board
推理执行以下命令
```text
sudo -E python3 bisenet_infer.py --backend_type rk_board \
                                  --model_path ./weights/rknn/bisenet.rknn 
```

## 踩坑

解决方案：不合适就换模型，换成最新的V2版本就可以了，但是上面的坑仍然存在