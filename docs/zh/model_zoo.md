目前paddle2onnx工具集主要支持转化的模型有三大类，图像分类，图像检测和图像分割。随着Paddle 2.0的开发，序列化算子的实现将更加通用，未来有望支持NLP, OCR系列模型的转换。
受限于不同框架的差异，部分模型可能会存在目前无法转换的情况，如若您发现无法转换或转换失败，或者转换后模型预测存在误差等问题，欢迎通过[ISSUE反馈](https://github.com/PaddlePaddle/paddle-onnx/issues/new)的方式告知我们(模型名，代码实现或模型获取方式)，我们会即时跟进：

# 动态图模型
目前paddle动态图模型还在开发中，随着Paddle动态图模型的增加，我们会及时更新可转换的动态图模型类型。

## 图像分类

图像分类模型比较完善，目前已支持paddlecls [dygraph分支](https://github.com/paddlepaddle/paddleclas/tree/dygraph)中全系列的模型。

|模型名称 | 来源 |  
|---|---|
| ResNet及其Vd系列 | [paddleclas](https://github.com/paddlepaddle/paddleclas/blob/dygraph/readme_cn.md#resnet%e5%8f%8a%e5%85%b6vd%e7%b3%bb%e5%88%97)|
| 移动端系列(MobileNet等)| [paddleclas](https://github.com/paddlepaddle/paddleclas/blob/dygraph/readme_cn.md#%e7%a7%bb%e5%8a%a8%e7%ab%af%e7%b3%bb%e5%88%97)|
| SEResNeXt与Res2Net系列 | [paddleclas](https://github.com/paddlepaddle/paddleclas/blob/dygraph/readme_cn.md#seresnext%e4%b8%8eres2net%e7%b3%bb%e5%88%97)|
| DPN与DenseNet系列 |[paddleclas](https://github.com/paddlepaddle/paddleclas/blob/dygraph/readme_cn.md#dpn%e4%b8%8edensenet%e7%b3%bb%e5%88%97)|
| HRNet系列|[paddleclas](https://github.com/paddlepaddle/paddleclas/blob/dygraph/readme_cn.md#hrnet%e7%b3%bb%e5%88%97)|
| Inception系列 |[paddleclas](https://github.com/PaddlePaddle/PaddleClas/blob/dygraph/README_cn.md#inception%E7%B3%BB%E5%88%97)|
| EfficientNet与ResNeXt101_wsl系列 |[paddleclas](https://github.com/paddlepaddle/paddleclas/blob/dygraph/readme_cn.md#efficientnet%e4%b8%8eresnext101_wsl%e7%b3%bb%e5%88%97)|
| ResNeSt与RegNet系列|[paddleclas](https://github.com/paddlepaddle/paddleclas/blob/dygraph/readme_cn.md#resnest%e4%b8%8eregnet%e7%b3%bb%e5%88%97)|
| Transformer系列 |[paddleclas](https://github.com/paddlepaddle/paddleclas/blob/dygraph/readme_cn.md#transformer%e7%b3%bb%e5%88%97)|
| 其他模型 |[paddleclas](https://github.com/paddlepaddle/paddleclas/blob/dygraph/readme_cn.md#%e5%85%b6%e4%bb%96%e6%a8%a1%e5%9e%8b)|

## 图像OCR
支持的模型有DB(文字检测)，CRNN(文字识别)，以及方向分类模型，测试模型来自于PaddleOCR [dgraph分支](https://github.com/PaddlePaddle/PaddleOCR//tree/dygraph)。

| 模型 | 来源 |
|-------|--------|
|DB|[PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR/blob/dygraph/doc/doc_ch/algorithm_overview.md#1%E6%96%87%E6%9C%AC%E6%A3%80%E6%B5%8B%E7%AE%97%E6%B3%95) |
|CRNN|[PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR/blob/dygraph/doc/doc_ch/algorithm_overview.md#2%E6%96%87%E6%9C%AC%E8%AF%86%E5%88%AB%E7%AE%97%E6%B3%95) |
|CLS|[PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR/blob/dygraph/doc/doc_ch/models_list.md#%E4%B8%89%E6%96%87%E6%9C%AC%E6%96%B9%E5%90%91%E5%88%86%E7%B1%BB%E6%A8%A1%E5%9E%8B) |

## 图像分割
支持的模型有UNet, HRNet, DeepLab等模型，测试模型来自于PaddleSeg [release/v2.0.0-rc分支](https://github.com/PaddlePaddle/PaddleSeg/tree/release/v2.0.0-rc)。

| 模型 | 来源 |
|-------|--------|
|BiSeNet|[PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg/tree/release/v2.0.0-rc/configs/bisenet) |
|DANet|[PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg/blob/release/v2.0.0-rc/configs/danet) |
|DeepLabv3|[PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg/blob/release/v2.0.0-rc/configs/deeplabv3) |
|Deeplabv3P |[PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg/blob/release/v2.0.0-rc/configs/deeplabv3p) |
|FCN|[PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg/blob/release/v2.0.0-rc/configs/fcn) |
|GCNet|[PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg/blob/release/v2.0.0-rc/configs/gcnet) |
|OCRNet|[PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg/blob/release/v2.0.0-rc/configs/ocrnet) |
|UNet|[PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg/blob/release/v2.0.0-rc/configs/unet) |

## 图像检测
待测试

## 自然语言处理
目前支持的模型有ERNIE系列模型，测试模型来自于PaddleNLP [2.0-beta 分支](https://github.com/PaddlePaddle/models/tree/release/2.0-beta/PaddleNLP)。

| 模型 | 来源 |
|-------|--------|
|ERNIE-1.0|[PaddleNLP](https://github.com/PaddlePaddle/models/blob/develop/PaddleNLP/docs/models.md#paddlenlpmodels) |
|ERNIE-2.0|[PaddleNLP](https://github.com/PaddlePaddle/models/blob/develop/PaddleNLP/docs/models.md#paddlenlpmodels) |

# 静态图模型
## 图像分类
图像分类模型支持比较完善，测试模型来自于 PaddleCls [master 分支](https://github.com/PaddlePaddle/PaddleClas/tree/master)。

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
支持的模型有SSD、YoloV3、FasterRCNN模型，测试模型来自于PaddleDetection [release/0.4分支](https://github.com/PaddlePaddle/Paddledetection/tree/release/0.4)。由于ONNX对检测模型算子支持比较有限，paddle2onnx对检测模型也不能完全支持。后续我们计划增加对其它检测模型的支持，基于ONNX目前对检测模型支持的现状，将会主要集中于一阶段检测模型。

| 模型 | 来源 |
|-------|--------|
|SSD_MobileNet|[PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection/blob/release/0.4/docs/MODEL_ZOO_cn.md#ssd) |
|YoloV3_DarkNet53|[PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection/blob/release/0.4/docs/MODEL_ZOO_cn.md#yolo-v3-%E5%9F%BA%E4%BA%8Epasacl-voc%E6%95%B0%E6%8D%AE%E9%9B%86) |
|YoloV3_ResNet34|[PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection/blob/release/0.4/docs/MODEL_ZOO_cn.md#yolo-v3-%E5%9F%BA%E4%BA%8Epasacl-voc%E6%95%B0%E6%8D%AE%E9%9B%86) |
|YoloV3_MobileNet|[PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection/blob/release/0.4/docs/MODEL_ZOO_cn.md#yolo-v3-%E5%9F%BA%E4%BA%8Epasacl-voc%E6%95%B0%E6%8D%AE%E9%9B%86) |
|FasterRCNN|[PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection/blob/release/0.4/docs/MODEL_ZOO_cn.md#faster--mask-r-cnn) |
|FasterRCNN_FPN|[PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection/blob/release/0.4/docs/MODEL_ZOO_cn.md#faster--mask-r-cnn) |
|FasterRCNN_FPN_DCN|[PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection/blob/release/0.4/docs/MODEL_ZOO_cn.md#deformable-%E5%8D%B7%E7%A7%AF%E7%BD%91%E7%BB%9Cv2) |

## 图像分割
支持的模型有UNet, HRNet, DeepLab模型，测试模型来自于PaddleSeg [release/v0.7.0分支](https://github.com/PaddlePaddle/PaddleSeg/tree/release/v0.7.0)

| 模型 | 来源 |
|-------|--------|
|UNet|[PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg/blob/release/v0.7.0/tutorial/finetune_unet.md) |
|HRNet|[PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg/blob/release/v0.7.0/tutorial/finetune_hrnet.md) |
|DeepLab|[PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg/blob/release/v0.7.0/tutorial/finetune_deeplabv3plus.md) |
