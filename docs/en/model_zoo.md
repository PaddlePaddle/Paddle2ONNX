Paddle2ONNX mainly supports three types of models: image classification, object detection and image segmentation.
As PaddlePaddle 2.0 evolves, the realization of serialized operaters will be more universal. It is expected that NLP and OCR series of models will be supported.

Due to the differences between frameworks, some models may not be supported. If you meet any problem such as converting failure or inference error，you can raise a issue in [ISSUE](https://github.com/PaddlePaddle/paddle-onnx/issues/new).

# Dynamic computational graph

As dynamic computational graph is under develop, test models are few and you can find them in [models.dygraph](https://github.com/PaddlePaddle/models/tree/release/1.8/dygraph). We will update more convertable models of dynamic computational graphs as the develop of PaddlePaddle.

|Models | Source |  
|---|---|
| MobileNetV1| [models.dygraph](https://github.com/PaddlePaddle/models/blob/f80b766295e4c686d3d6d00858656d8239cea87f/dygraph/mobilenet/mobilenet_v1.py#L106)|
| MobileNetV2| [model.dygraph](https://github.com/PaddlePaddle/models/blob/f80b766295e4c686d3d6d00858656d8239cea87f/dygraph/mobilenet/mobilenet_v2.py#L153)|
| ResNet| [models.dygraph](https://github.com/PaddlePaddle/models/blob/release/1.8/dygraph/resnet/train.py#L170)|
| Mnist|[models.dygraph](https://github.com/PaddlePaddle/models/blob/f80b766295e4c686d3d6d00858656d8239cea87f/dygraph/mnist/train.py#L89)|

# Static computational graph
## Image classification
Test models are from [PaddleCls](https://github.com/PaddlePaddle/PaddleClas).

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

## Object Detection
Support SSD,YoloV3. Test models are from [PaddleDetection](https://github.com/PaddlePaddle/Paddledetection).
Due to ONNX's limit, Paddle2ONNX is not able to support all the detection models and now  only supports one-stage detection model.

| Models | Source |
|-------|--------|
|SSD_MobileNet|[PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection/blob/release/0.4/docs/MODEL_ZOO_cn.md#ssd) |
|YoloV3_DarkNet53|[PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection/blob/release/0.4/docs/MODEL_ZOO_cn.md#yolo-v3-%E5%9F%BA%E4%BA%8Epasacl-voc%E6%95%B0%E6%8D%AE%E9%9B%86) |
|YoloV3_ResNet34|[PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection/blob/release/0.4/docs/MODEL_ZOO_cn.md#yolo-v3-%E5%9F%BA%E4%BA%8Epasacl-voc%E6%95%B0%E6%8D%AE%E9%9B%86) |
|YoloV3_MobileNet|[PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection/blob/release/0.4/docs/MODEL_ZOO_cn.md#yolo-v3-%E5%9F%BA%E4%BA%8Epasacl-voc%E6%95%B0%E6%8D%AE%E9%9B%86) |
|FasterRCNN|[PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection/blob/release/0.4/docs/MODEL_ZOO_cn.md#faster--mask-r-cnn) |

## Image segmentation
Support UNet,HRNet and DeepLab. Test models are from [PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg).

| Models | Source |
|-------|--------|
|UNet|[PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg/blob/release/v0.7.0/tutorial/finetune_unet.md) |
|HRNet|[PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg/blob/release/v0.7.0/tutorial/finetune_hrnet.md) |
|DeepLab|[PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg/blob/release/v0.7.0/tutorial/finetune_deeplabv3plus.md) |

## OCR

| Models | Source |
|-------|--------|
|DB|[PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR/blob/develop/doc/doc_en/algorithm_overview_en.md#1-text-detection-algorithm) |
