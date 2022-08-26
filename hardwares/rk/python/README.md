# Paddle 转 RK部署环境搭建
RK的部署环境有两种：
1. PC环境中搭建的RK模型运行和模型转换环境，用于模拟RK推理和模型转换等工作
2. RK部署平台上的环境，一般由芯片提供商提供，用于部署实际任务
本文档介绍在PC上搭建RK系列芯片的模拟运行及模型转换环境

## 软件栈要求
OS：Ubuntu18.04  
Python版本：Python3.8
Paddle2ONNX版本：paddle2onnx-0.9.8
PaddlePaddle：PaddlePaddle-gpu develop版  
rknn-toolkit2：version 1.3.4b19   链接：https://eyun.baidu.com/s/3eTDMk6Y 密码：rknn
## 安装步骤
### 模型转换相关
```
python -m pip install paddlepaddle-gpu==0.0.0.post102 -f https://www.paddlepaddle.org.cn/whl/linux/gpu/develop.html
pip install onnx==1.7.0
pip install onnxruntime
pip install paddle2onnx
```

### 安装RK所需库
#### rknn-toolkit2安装踩坑(for PC)

- 下载需要的软件包

```text
sudo apt-get install libxslt1-dev zlib1g zlib1g-dev libglib2.0-0 \
libsm6 libgl1-mesa-glx libprotobuf-dev gcc g++
```
- 安装rknn-toolkit2过程中提前要安装的包

```text
pip install numpy==1.16.6
```
- 安装rknn-toolkit2

```text
pip install rknn_toolkit2-1.3.0_11912b58-cp38-cp38-linux_x86_64.whl
```

#### rknn_toolkit_lite2安装踩坑(for Board)

##### 下载必备的资源

- **下载，并将包传到板子上**
    - 进入[RK仓库](https://github.com/rockchip-linux/rknn-toolkit2)并且下载rknn-toolkit2
    - 进入[RK仓库](https://github.com/rockchip-linux/rknpu2)并且下载rknpu2
    - 解压并将rknn_toolkit_lite2，rknpu2-master传到板子上，我这里传到Download目录下

```text
toybrick@debian10:~/Downloads$ ls
rknn_toolkit_lite2  rknpu2-master
```

##### 安装各种包

- **安装RKNN Toolkit Lite2**

目前可以通过 pip3 install 命令安装 RKNN Toolkit Lite2。

注: 以下安装过程大部分来自RK官方文档，做了一些修改

如果系统中没有安装 python3/pip3 等程序，请先通过 apt-get 方式安装，命令如下:

```text
sudo apt update
sudo apt-get install -y python3 python3-dev python3-pip gcc
```
  **切记这里不是sudo apt-get update!!!**
- **安装依赖模块: opencv-python 和 numpy**

```text
sudo apt-get install -y python3-opencv 
sudo apt-get install -y python3-numpy
```
- **安装 RKNN Toolkit Lite2**

```text
cd ~/Download/rknn_toolkit_lite2/packages/
pip3 install rknn_toolkit_lite2-1.3.0-cp37-cp37m-linux_aarch64.whl
```

##### 安装RKNPU2 Linux 驱动

安装完RKNN Toolkit Lite2，我们还需要安装RKNPU2驱动

```text
cd ~/Downloads/rknpu2-master/runtime/RK3588/Linux/librknn_api/aarch64/
sudo cp ./* /usr/lib

```

##### 运行example

安装完毕后，我们需要进行测试

```text
cd ~/Downloads/rknn_toolkit_lite2/examples/inference_with_lite/
sudo -E python3 test.py
```

输出

```text
--> Load RKNN model
done
--> Init runtime environment
I RKNN: [09:38:10.438] RKNN Runtime Information: librknnrt version: 1.3.0 (9b36d4d74@2022-05-04T20:17:01)
I RKNN: [09:38:10.439] RKNN Driver Information: version: 0.4.2
I RKNN: [09:38:10.439] RKNN Model Information: version: 1, toolkit version: 1.3.0-11912b58(compiler version: 1.3.0 (9b36d4d74@2022-05-04T20:21:47)), target: RKNPU lite, target platform: rk3568, framework name: PyTorch, framework layout: NCHW
done
--> Running model
resnet18
-----TOP 5-----
[812]: 0.9996383190155029
[404]: 0.00028062646742910147
[657]: 1.632110434002243e-05
[833 895]: 1.015904672385659e-05
[833 895]: 1.015904672385659e-05

done
```
## 注意事项
1. RK依赖安装参考：[RK文档](https://github.com/rockchip-linux/rknn-toolkit2/blob/master/doc/Rockchip_Quick_Start_RKNN_Toolkit2_CN-1.2.0.pdf)  
2. PaddlePaddle安装参考：[Paddle安装文档](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/develop/install/pip/linux-pip.html)  
3. 分类模型部署请参考：[分类模型部署](./classification/README.md)
