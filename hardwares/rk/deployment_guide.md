# RK模型部署指南
本文档介绍在RK系列芯片上部署Paddle模型的步骤，具体包括：  
1. 在Ubuntu18.06机器上安装RK模型运行环境和Paddle2ONNX等运行环境。  
2. 使用Paddle2ONNX将PaddleInference model转换为ONNX模型格式。
3. 运行推理脚本获得推理结果。

## 环境准备
PC环境要求:
OS：Ubuntu18.04  
Python版本：Python3.6  
```
#安装Paddle2ONNX
git clone https://github.com/PaddlePaddle/Paddle2ONNX.git
cd Paddle2ONNX
python setup.py install

# 安装onnxruntime
pip install onnxruntime

#安装RK所需库
sudo apt-get install libxslt1-dev zlib1g-dev libglib2.0-0 libgl1-mesa-glx libsm6 libprotobuf-dev
sudo apt-get install python3 python3-dev python3-pip
git clone https://github.com/rockchip-linux/rknn-toolkit2
cd rknn-toolkit2
python -m pip install -r doc/requirements*.txt
cd pakage
python -m pip install rknn_toolkit2*.whl
```
RK依赖安装参考：[RK文档](https://github.com/rockchip-linux/rknn-toolkit2/blob/master/doc/Rockchip_Quick_Start_RKNN_Toolkit2_CN-1.2.0.pdf)

## 部署指导
部署分为以下两个步骤：  
1. Paddle模型转换为ONNX模型
2. 加载ONNX模型进行推理
### 模型转换
RK尚不支持动态shape的输入，因此在使用Paddle2ONNX将Paddle模型转换为ONNX模型格式时需要输入input_shape
接下来以mobilenetv3作为转换示例
```
# 下载mobilenetv3模型
wget https://bj.bcebos.com/paddle2onnx/model_zoo/mobilenetv3.tar.gz
tar xvf mobilenetv3.tar.gz
# 将Paddle模型导出为ONNX模型
paddle2onnx --model_dir ./ --model_filename inference.pdmodel --params_filename inference.pdiparams --save_file mobilenetv3.onnx --opset_version 12 --enable_onnx_checker True  --input_shape_dict "{'inputs': [1, 3, 224, 224]}"
```
## ONNX模型推理示例
以mobilenetv3分类模型为例
### 使用RK进行推理
```
python deploy.py --model_file mobilenetv3.onnx --image_path images/ILSVRC2012_val_00000010.jpeg --backend_type rk
```
RK推理结果：![图片](./images/doc_imgs/class_rk.png)  
### 使用ONNXRuntime进行推理
```
# 使用ONNXRuntime进行推理
python deploy.py --model_file mobilenetv3.onnx --image_path images/ILSVRC2012_val_00000010.jpeg --backend_type onnxruntime
```
ONNXRuntime推理结果：![图片](./images/doc_imgs/class_onnxruntime.png)
### 注意事项
> 各类别id与明文标签参考[ImageNet标签](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.3/deploy/utils/imagenet1k_label_list.txt)
> RK相关API可参考文档： [RK API文档](https://github.com/rockchip-linux/rknn-toolkit2/blob/master/doc/Rockchip_User_Guide_RKNN_Toolkit2_CN-1.2.0.pdf)  
> ONNXRuntime要求输入为NCHW，RK要求输入为NHWC
