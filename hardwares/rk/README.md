# RK部署环境搭建
本文档介绍在PC上搭建RK系列芯片的模型运行及模型转换环境

## 软件栈要求
OS：Ubuntu18.04  
Python版本：Python3.6  
Paddle2ONNX版本：develop版  
PaddlePaddle：PaddlePaddle-gpu develop版  
rknn-toolkit2：version 1.2.0  
## 安装步骤
### 模型转换相关
```
python -m pip install paddlepaddle-gpu==0.0.0.post102 -f https://www.paddlepaddle.org.cn/whl/linux/gpu/develop.html
git clone https://github.com/PaddlePaddle/Paddle2ONNX.git
cd Paddle2ONNX
python setup.py install
python -m pip install onnxruntime
```

### 安装RK所需库
```
sudo apt-get install libxslt1-dev zlib1g-dev libglib2.0-0 libgl1-mesa-glx libsm6 libprotobuf-dev
sudo apt-get install python3 python3-dev python3-pip
git clone https://github.com/rockchip-linux/rknn-toolkit2
cd rknn-toolkit2
python -m pip install -r doc/requirements*.txt
cd packages
python -m pip install rknn_toolkit2*.whl
```
## 注意事项
1. RK依赖安装参考：[RK文档](https://github.com/rockchip-linux/rknn-toolkit2/blob/master/doc/Rockchip_Quick_Start_RKNN_Toolkit2_CN-1.2.0.pdf)  
2. PaddlePaddle安装参考：[Paddle安装文档](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/develop/install/pip/linux-pip.html)  
3. 分类模型部署请参考：[分类模型部署](./classification/README.md)
