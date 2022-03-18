# RK模型分类模型部署指南
本文档介绍在RK系列芯片上部署Paddle分类模型的步骤，具体包括：  
1. 使用Paddle2ONNX将PaddleInference model转换为ONNX模型格式。
2. 运行推理脚本获得推理结果。  

**RK PC模拟环境快速部署脚本**：bash quick_deploy.sh
## 模型转换
以mobilenetv3作为转换示例
```
# 下载mobilenetv3模型
wget https://bj.bcebos.com/paddle2onnx/model_zoo/mobilenetv3.tar.gz
tar xvf mobilenetv3.tar.gz
# 将Paddle模型导出为ONNX模型
paddle2onnx --model_dir ./mobilenetv3 --model_filename inference.pdmodel --params_filename inference.pdiparams --save_file mobilenetv3.onnx --opset_version 12 --enable_onnx_checker True  --input_shape_dict "{'inputs': [1, 3, 224, 224]}"
```
## 模型推理示例
以mobilenetv3分类模型为例
### 在RK PC模拟环境中进行推理
```
python deploy.py --model_file mobilenetv3.onnx --image_path images/ILSVRC2012_val_00000010.jpeg --backend_type rk_pc

# 运行结果
TopK Indices:  [153 283 204 259 265]
TopK Scores:  [0.5834961  0.14819336 0.02505493 0.01279449 0.01192474]
```
### 在RK部署平台上进行推理
在RK部署平台上部署推理需要进行三个简单步骤：
1. 在RK PC环境中运行推理，生成RK硬件上运行的rknn模型
2. 将模型的部署脚本拷贝到RK部署平台上
3. 运行部署程序
```
# step 1: 生成部署模型，模型后缀rknn
python deploy.py --model_file mobilenetv3.onnx --image_path images/ILSVRC2012_val_00000010.jpeg --backend_type rk_pc

# step 2：拷贝模型和脚本到部署平台
scp -r * xxx@ip:folder

# step 3：在部署平台上运行部署脚本
python3 deploy.py --model_file model.rknn --image_path images/ILSVRC2012_val_00000010.jpeg --backend_type rk_hardware

# 运行结果
TopK Indices:  [153 283 204 259 265]
TopK Scores:  [0.5917969  0.14221191 0.02580261 0.01242828 0.01228333]
```
### 使用ONNXRuntime进行推理
```
# 使用ONNXRuntime进行推理
python deploy.py --model_file mobilenetv3.onnx --image_path images/ILSVRC2012_val_00000010.jpeg --backend_type onnxruntime

# 运行结果
TopK Indices:  [153 283 204 259 265]
TopK Scores:  [0.5918329  0.1443437  0.02467788 0.01226414 0.01210706]
```
### 使用Paddle进行推理
```
# 使用Paddle进行推理
python deploy.py --image_path images/ILSVRC2012_val_00000010.jpeg --backend_type paddle --model_dir mobilenetv3

# 运行结果
TopK Indices:  [153 283 204 259 265]
TopK Scores:  [0.59183234 0.14434433 0.02467804 0.01226414 0.01210703]
```
## 注意事项
1. 各类别id与明文标签参考[ImageNet标签](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.3/deploy/utils/imagenet1k_label_list.txt)
2. RK尚不支持动态shape的输入，因此在使用Paddle2ONNX将Paddle模型转换为ONNX模型格式时需要输入input_shape
3. RK支持Opset version <= 12
4. RK相关API可参考文档： [RK API文档](https://github.com/rockchip-linux/rknn-toolkit2/blob/master/doc/Rockchip_User_Guide_RKNN_Toolkit2_CN-1.2.0.pdf)  
5. ONNXRuntime要求输入为NCHW，RK要求输入为NHWC
6. RK的PC模拟环境中，backend_type设置为rk_pc
7. 在RK部署芯片上，backend_type设置为rk_hardware
