# Bisenet c++部署
部署在c++上之前，请先运行rknpu2/python代码，将onnx模型转换成rknn模型

## 下载模型
```text
mkdir weights
cd weights
wget https://paddlelite-demo.bj.bcebos.com/onnx%26rknn2_model/bisenet.rknn
```

## 编译代码
```text
sh ./build-linux_RK3588.sh
```

## 运行代码
```text
# 进入文件夹
cd ./install/Bisenet_demo_Linux

# 设置库文件路径
export LD_LIBRARY_PATH=./lib

# 运行代码
sudo ./Bisenet_demo
```

## 查看结果
PP_OCR_det模型结构展示
输入图片
![输入图片](./install/Bisenet_demo_Linux/images/before/bisenet_demo_input.jpeg)

输出图片
![输出图片](./install/Bisenet_demo_Linux/images/after/results.jpg)

## 推理速度
取三次平均后为61.32ms，比python大概快1/7，用的一样的api不知道为啥c++就快点