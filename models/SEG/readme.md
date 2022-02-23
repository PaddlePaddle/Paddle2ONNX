# 图像分割模型库

本文档中模型库均来源于[PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg)，在下表中提供了部分已经转换好的模型，如有更多模型或自行模型训练导出需求，可参考[PaddleSeg使用文档]().
|模型名称|配置文件|模型大小|下载地址| 说明 |
| --- | --- | --- | --- | ---- |
|BiSeNet|[bisenet_cityscapes_1024x1024_160k.yml](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.4/configs/bisenet/bisenet_cityscapes_1024x1024_160k.yml)| 3M |[推理模型]() / [ONNX模型](model.onnx)| CityScape数据训练数据，80个分类，包括车、路、人等等 |
|DANet|[danet_resnet50_os8_cityscapes_1024x512_80k.yml](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.4/configs/danet/danet_resnet50_os8_cityscapes_1024x512_80k.yml)|3M|[推理模型]() / [ONNX模型](model.onnx)|
|DeepLabv3|[deeplabv3_resnet50_os8_cityscapes_1024x512_80k.yml](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.4/configs/deeplabv3/deeplabv3_resnet50_os8_cityscapes_1024x512_80k.yml)| 2.6M |[推理模型]() / [ONNX模型](model.onnx)|
|Deeplabv3P|[deeplabv3p_resnet50_os8_cityscapes_1024x512_80k.yml](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.4/configs/deeplabv3p/deeplabv3p_resnet50_os8_cityscapes_1024x512_80k.yml)|3M|[推理模型]() / [ONNX模型](model.onnx)|
|FCN|[fcn_hrnetw18_cityscapes_1024x512_80k.yml](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.4/configs/fcn/fcn_hrnetw18_cityscapes_1024x512_80k.yml)|47M|[推理模型]() / [ONNX模型](model.onnx)|

## 使用ONNXRuntime加载预测
```
pip install paddlepaddle
pip install onnxruntime
```
下载模型
```
wget xxx
```
在本目录下，我们提供了`infer.py`和`demo.jpg`进行预测，执行如下命令即可
```
python infer.py xxxx
```
