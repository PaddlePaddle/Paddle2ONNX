# PP_HumanSeg c++部署
部署在c++上之前，请先运行rknpu2/python代码，将onnx模型转换成rknn模型

## 下载权重文件
```text
mkdir model
cd model
wget https://paddlelite-demo.bj.bcebos.com/onnx%26rknn2_model/portrait_pp_humansegv2_lite_256x144_inference_model_with_softmax.rknn
```

## 编译代码
```text
sh ./build-linux_RK3588.sh
```

## 运行代码
```text
# 进入文件夹
cd ./install/PP_HumanSeg_demo_Linux

# 设置库文件路径
export LD_LIBRARY_PATH=./lib

# 运行代码
sudo ./PP_HumanSeg_demo
```

## 查看结果
输入图片
![输入图片](./install/PP_HumanSeg_demo_Linux/images/before/PP_HumanSeg_demo_input.jpg)

输出图片
![输出图片](./install/PP_HumanSeg_demo_Linux/images/after/result.jpg)

## 推理速度
取三次平均后为61.32ms，比python大概快1/7，用的一样的api不知道为啥c++就快点