目前paddle2onnx工具集主要支持转化的模型有三大类，图像分类，图像检测和图像分割。随着Paddle 2.0的开发，序列化算子的实现将更加通用，未来有望支持NLP, OCR系列模型的转换。
受限于不同框架的差异，部分模型可能会存在目前无法转换的情况，如若您发现无法转换或转换失败，或者转换后模型预测存在误差等问题，欢迎通过[ISSUE反馈](https://github.com/PaddlePaddle/paddle-onnx/issues/new)的方式告知我们(模型名，代码实现或模型获取方式)，我们会即时跟进：

# 动态图模型
目前paddle动态图模型还在开发中，可稳定进行测试的模型较少，测试模型来自于[models.dygraph](https://github.com/PaddlePaddle/models/tree/release/1.8/dygraph)。随着Paddle动态图模型的增加，我们会及时更新可转换的动态图模型类型。

|模型名称 | 来源 |  
|---|---|
| MobileNetV1| [models.dygraph](https://github.com/PaddlePaddle/models/blob/f80b766295e4c686d3d6d00858656d8239cea87f/dygraph/mobilenet/mobilenet_v1.py#L106)|
| MobileNetV2| [model.dygraph](https://github.com/PaddlePaddle/models/blob/f80b766295e4c686d3d6d00858656d8239cea87f/dygraph/mobilenet/mobilenet_v2.py#L153)|
| ResNet| [models.dygraph](https://github.com/PaddlePaddle/models/blob/release/1.8/dygraph/resnet/train.py#L170)|
| Mnist|[models.dygraph](https://github.com/PaddlePaddle/models/blob/f80b766295e4c686d3d6d00858656d8239cea87f/dygraph/mnist/train.py#L89)|

# 静态图模型
## 图像分类
图像分类模型支持比较完善，测试模型来自于[PaddleCls](https://github.com/PaddlePaddle/PaddleClas)。

| 模型 | 来源 |
|-------|--------|
| ResNet | [PaddleCls](https://github.com/PaddlePaddle/PaddleClas/blob/master/ppcls/modeling/architectures/resnet.py) |
| DenseNet | [PaddleClas](https://github.com/PaddlePaddle/PaddleClas/blob/master/ppcls/modeling/architectures/densenet.py) |
| ShuffleNet | [PaddleCls](https://github.com/PaddlePaddle/PaddleClas/blob/master/ppcls/modeling/architectures/shufflenet_v2.py) |
| MobileNet| [PaddleCls](https://github.com/PaddlePaddle/PaddleClas/blob/master/ppcls/modeling/architectures/mobilenet_v3.py) |
| VGG16| [PaddleCls](https://github.com/PaddlePaddle/PaddleClas/blob/master/ppcls/modeling/architectures/vgg.py) |
| SE_ResNext50| [PaddleCls](https://github.com/PaddlePaddle/PaddleClas/blob/master/ppcls/modeling/architectures/se_resnext.py) |
| InceptionV4| [PaddleCls](https://github.com/PaddlePaddle/PaddleClas/blob/master/ppcls/modeling/architectures/inception_v4.py) |
| SE_ResNet50_vd| [PaddleCls](https://github.com/PaddlePaddle/PaddleClas/blob/master/ppcls/modeling/architectures/se_resnext_vd.py) |
| SqueezeNet1_0| [PaddleCls](https://github.com/PaddlePaddle/PaddleClas/blob/master/ppcls/modeling/architectures/squeezenet.py) |
| Xception41| [PaddleCls](https://github.com/PaddlePaddle/PaddleClas/blob/master/ppcls/modeling/architectures/xception.py) |

## 图像检测
支持的模型有SSD、YoloV3模型，测试模型来自于[PaddleDetection](https://github.com/PaddlePaddle/Paddledetection)。由于ONNX对检测模型算子支持比较有限，paddle2onnx对检测模型也不能完全支持。后续我们计划增加对其它检测模型的支持，基于ONNX目前对检测模型支持的现状，将会集中于一阶段检测模型。

| 模型 | 来源 |
|-------|--------|
|SSD_MobileNet|[PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection/blob/release/0.4/docs/MODEL_ZOO_cn.md#ssdlite) |
|YoloV3_DarkNet53|[PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection/blob/release/0.4/docs/MODEL_ZOO_cn.md#yolo-v3-%E5%9F%BA%E4%BA%8Epasacl-voc%E6%95%B0%E6%8D%AE%E9%9B%86) |
|YoloV3_ResNet34|[PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection/blob/release/0.4/docs/MODEL_ZOO_cn.md#yolo-v3-%E5%9F%BA%E4%BA%8Epasacl-voc%E6%95%B0%E6%8D%AE%E9%9B%86) |
|YoloV3_MobileNet|[PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection/blob/release/0.4/docs/MODEL_ZOO_cn.md#yolo-v3-%E5%9F%BA%E4%BA%8Epasacl-voc%E6%95%B0%E6%8D%AE%E9%9B%86) |

## 图像分割
支持的模型有UNet, HRNet, DeepLab模型，测试模型来自于[PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg)。

| 模型 | 来源 |
|-------|--------|
|UNet|[PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg/blob/release/v0.7.0/tutorial/finetune_unet.md) |
|HRNet|[PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg/blob/release/v0.7.0/tutorial/finetune_hrnet.md) |
|DeepLab|[PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg/blob/release/v0.7.0/tutorial/finetune_deeplabv3plus.md) |

## 图像OCR
支持的模型有UNet, HRNet, DeepLab模型，测试模型来自于[PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg)。

| 模型 | 来源 |
|-------|--------|
|DB|[PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR/blob/develop/doc/doc_en/algorithm_overview_en.md#1-text-detection-algorithm) |
