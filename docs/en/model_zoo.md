Paddle2ONNX mainly supports three types of models: image classification, object detection and image segmentation.
As PaddlePaddle 2.0 evolves, the realization of serialized operators will be more universal. It is expected that NLP and OCR series of models will be supported.

Due to the differences between frameworks, some models may not be supported. If you meet any problem such as converting failure or inference error，you can raise a issue in [ISSUE](https://github.com/PaddlePaddle/paddle-onnx/issues/new).

# Dynamic computational graph

As dynamic computational model is under develop,  We will update more convertable models of dynamic computational graphs as the develop of PaddlePaddle.

## Image classification

Comprehensive coverage of image classification models，now we support the whole series model in PaddlClas  [dygraph branch](https://github.com/paddlepaddle/PaddleClas/tree/dygraph).

|Models | Source |  
|---|---|
| ResNet series| [PaddleClas](https://github.com/PaddlePaddle/PaddleClas/blob/dygraph/README.md#resnet-and-vd-series)|
| Mobile series | [PaddleClas](https://github.com/PaddlePaddle/PaddleClas/blob/dygraph/README.md#mobile-series)|
| SEResNeXt and Res2Net series | [PaddleClas](https://github.com/PaddlePaddle/PaddleClas/blob/dygraph/README.md#seresnext-and-res2net-series)|
| DPN and DenseNet series |[PaddleClas](https://github.com/PaddlePaddle/PaddleClas/blob/dygraph/README.md#dpn-and-densenet-series)|
| HRNet series |[PaddleClas](https://github.com/PaddlePaddle/PaddleClas/blob/dygraph/README.md#hrnet-series)|
| Inception series |[PaddleClas](https://github.com/PaddlePaddle/PaddleClas/blob/dygraph/README.md#inception-series)|
| EfficientNet and ResNeXt101_wsl series |[PaddleClas](https://github.com/PaddlePaddle/PaddleClas/blob/dygraph/README.md#efficientnet-and-resnext101_wsl-series)|
| ResNeSt and RegNet series |[PaddleClas](https://github.com/PaddlePaddle/PaddleClas/blob/dygraph/README.md#resnest-and-regnet-series)|
| Transformer Series |[PaddleClas](https://github.com/PaddlePaddle/PaddleClas/blob/dygraph/README.md#transformer-series)|
| Others |[PaddleClas](https://github.com/PaddlePaddle/PaddleClas/blob/dygraph/README.md#others)|


## OCR
Support CRNN(Text Detection Model), DB(Text Recognition Model) and Text Angle Classification Model. Test models are form PaddleOCR [dygraph branch](https://github.com/PaddlePaddle/PaddleOCR//tree/dygraph).

| Models | Source |
|-------|--------|
|DB|[PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR/blob/develop/doc/doc_en/algorithm_overview_en.md#1-text-detection-algorithm) |
|CRNN|[PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR/blob/dygraph/doc/doc_en/algorithm_overview_en.md#2-text-recognition-algorithm) |
|CLS|[PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR/blob/dygraph/doc/doc_en/models_list_en.md#3-text-angle-classification-model) |

## Image segmentation
Support UNet, HRNet, DeepLab and so on. Test models are from PaddleSeg [release/v2.0.0-rc branch](https://github.com/PaddlePaddle/PaddleSeg/tree/release/v2.0.0-rc)。

| Models | Source |
|-------|--------|
|BiSeNet|[PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg/tree/release/v2.0.0-rc/configs/bisenet) |
|DANet|[PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg/blob/release/v2.0.0-rc/configs/danet) |
|DeepLabv3|[PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg/blob/release/v2.0.0-rc/configs/deeplabv3) |
|Deeplabv3P |[PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg/blob/release/v2.0.0-rc/configs/deeplabv3p) |
|FCN|[PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg/blob/release/v2.0.0-rc/configs/fcn) |
|GCNet|[PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg/blob/release/v2.0.0-rc/configs/gcnet) |
|OCRNet|[PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg/blob/release/v2.0.0-rc/configs/ocrnet) |
|UNet|[PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg/blob/release/v2.0.0-rc/configs/unet) |

## object detection
comming soon

# Static computational graph
## Image classification
Test models are from PaddleClas [master branch](https://github.com/PaddlePaddle/PaddleClas/tree/master).

| Models | Source |
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

## object detection
Support SSD,YoloV3. Test models are from PaddleDetection [release/0.4 branch](https://github.com/PaddlePaddle/Paddledetection/tree/release/0.4).
Due to ONNX's limit, Paddle2ONNX is not able to support all the detection models and now  only supports one-stage detection model.

| Models | Source |
|-------|--------|
|SSD_MobileNet|[PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection/blob/release/0.4/docs/MODEL_ZOO.md#ssd) |
|YoloV3_DarkNet53|[PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection/blob/release/0.4/docs/MODEL_ZOO.md#yolo-v3) |
|YoloV3_ResNet34|[PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection/blob/release/0.4/docs/MODEL_ZOO.md#yolo-v3) |
|YoloV3_MobileNet|[PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection/blob/release/0.4/docs/MODEL_ZOO.md#yolo-v3) |
|FasterRCNN|[PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection/blob/release/0.4/docs/MODEL_ZOO.md#faster--mask-r-cnn) |
|FasterRCNN_FPN|[PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection/blob/release/0.4/docs/MODEL_ZOO.md#faster--mask-r-cnn) |
|FasterRCNN_FPN_DCN|[PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection/blob/release/0.4/docs/MODEL_ZOO.md#deformable-convnets-v2) |

## Image segmentation
Support UNet,HRNet and DeepLab. Test models are from PaddleSeg [release/v0.7.0 branch](https://github.com/PaddlePaddle/PaddleSeg/tree/release/v0.7.0).

| Models | Source |
|-------|--------|
|UNet|[PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg/blob/release/v0.7.0/tutorial/finetune_unet.md) |
|HRNet|[PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg/blob/release/v0.7.0/tutorial/finetune_hrnet.md) |
|DeepLab|[PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg/blob/release/v0.7.0/tutorial/finetune_deeplabv3plus.md) |
