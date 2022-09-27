# PP_OCR_V2 c++部署
部署在c++上之前，请先运行rknpu2/python代码，将onnx模型转换成rknn模型

## 下载模型
```text
# 下载RKNN模型
cd model
wget https://paddlelite-demo.bj.bcebos.com/onnx%26rknn2_model/PP_OCR_v2_cls.rknn
wget https://paddlelite-demo.bj.bcebos.com/onnx%26rknn2_model/PP_OCR_v2_det.rknn
wget https://paddlelite-demo.bj.bcebos.com/onnx%26rknn2_model/PP_OCR_v2_rec.rknn
```

## 编译代码
```text
sh ./build-linux_RK3588.sh
```

## 运行代码
```text
# 进入文件夹
cd ./install/PP_OCR_demo_Linux

# 设置库文件路径
export LD_LIBRARY_PATH=./lib

# 运行代码
sudo ./PP_OCR_demo
```

## 查看结果
PP_OCR_det模型结构展示
输入图片
![输入图片](./install/PP_OCR_demo_Linux/images/before/lite_demo_input.png)

输出信息：
```text
-> Loading model
12: 0.996826
13: 0.999023
11: 0.997803
10: 0.993164
14: 0.999268
一: 0.864746
0.97417: 0.986956
0.995007: 0.990295
0.925898: 0.939392
0.985133: 0.956299
0.995842: 0.960144
0.996577: 0.989685
0.997668: 0.990417
0.961928: 0.990540
0.972573: 0.987610
0.990198: 0.991394
发足够的滋养: 0.980876
0.994448: 0.986938
（成品包材）: 0.897298
花费了0.457335秒: 0.969157
【净含量】：220ml: 0.959650
产品信息/参数0.992728: 0.964355
【品名】：纯臻营养护发素: 0.949666
纯臻营养护发素0.993604: 0.985937
【适用人群】：适合所有肤质: 0.927171
(45元／每公斤，100公斤起订）: 0.866757
【品牌】：代加工方式/0EMODM: 0.936006
糖、椰油酰胺丙基甜菜碱、泛醒: 0.918317
【产品编号】：YM-X-3011 0.96899: 0.947347
每瓶22元，1000瓶起订）0.993976: 0.977583
【主要成分】：鲸蜡硬脂醇、蒸麦B-葡聚: 0.900031
【主要功能】：可紧致头发磷层，从而达到: 0.944002
即时持久改善头发光泽的效果，给干燥的头: 0.990440
The detection visuaLized image saved in ./vis.jpg: 0.938965
```

## 推理速度
取三次平均后为61.32ms，比python大概快1/7，用的一样的api不知道为啥c++就快点