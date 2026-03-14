Paddle2ONNX supports converting PaddlePaddle model to ONNX format.

Due to the differences between frameworks, some models may not be supported. If you meet any problem such as converting failure or inference error，you can raise a issue in [ISSUE](https://github.com/PaddlePaddle/paddle-onnx/issues/new).


## Image classification

Comprehensive coverage of image classification models，now we support the whole series model in PaddlClas  [develop](https://github.com/PaddlePaddle/PaddleClas/tree/develop).

|Models | Source |
|---|---|
| ResNet series| [PaddleClas](https://github.com/PaddlePaddle/PaddleClas/tree/develop#ResNet_and_Vd_series)|
| Mobile series | [PaddleClas](https://github.com/PaddlePaddle/PaddleClas/tree/develop#Mobile_series)|
| SEResNeXt and Res2Net series | [PaddleClas](https://github.com/PaddlePaddle/PaddleClas/tree/develop#SEResNeXt_and_Res2Net_series)|
| DPN and DenseNet series |[PaddleClas](https://github.com/PaddlePaddle/PaddleClas/tree/develop#DPN_and_DenseNet_series)|
| HRNet series |[PaddleClas](https://github.com/PaddlePaddle/PaddleClas/tree/develop#HRNet_series)|
| Inception series |[PaddleClas](https://github.com/PaddlePaddle/PaddleClas/tree/develop#Inception_series)|
| EfficientNet and ResNeXt101_wsl series |[PaddleClas](https://github.com/PaddlePaddle/PaddleClas/tree/develop#EfficientNet_and_ResNeXt101_wsl_series)|
| ResNeSt and RegNet series |[PaddleClas](https://github.com/PaddlePaddle/PaddleClas/tree/develop#ResNeSt_and_RegNet_series)|


## OCR
Support CRNN(Text Detection Model), DB(Text Recognition Model) and Text Angle Classification Model. Test models are form PaddleOCR [dygraph branch](https://github.com/PaddlePaddle/PaddleOCR//tree/dygraph).

| Models | Source |
|-------|--------|
|Chinese and English ultra-lightweight OCR model (9.4M) |[PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR#pp-ocr-20-series-model-listupdate-on-dec-15) |
|Chinese and English general OCR model (143.4M)|[PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR#pp-ocr-20-series-model-listupdate-on-dec-15) |

## Image segmentation
Support UNet, HRNet, DeepLab and so on. Test models are from PaddleSeg [release/v2.0.0-rc branch](https://github.com/PaddlePaddle/PaddleSeg/tree/release/v2.0.0-rc)。

| Models | Source |
|-------|--------|
|BiSeNet|[PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg/tree/develop/configs/bisenet) |
|DANet|[PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg/blob/develop/configs/danet) |
|DeepLabv3|[PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg/blob/develop/configs/deeplabv3) |
|Deeplabv3P |[PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg/blob/develop/configs/deeplabv3p) |
|FCN|[PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg/blob/develop/configs/fcn) |
|GCNet|[PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg/blob/develop/configs/gcnet) |
|OCRNet|[PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg/blob/develop/configs/ocrnet) |
|UNet|[PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg/blob/develop/configs/unet) |

## object detection
Support 8 object detection archtectures. Test models are from PaddleDetection [develop](https://github.com/PaddlePaddle/PaddleDetection/tree/develop)
| Models      | Source                                                       |
| ----------- | ------------------------------------------------------------ |
| YOLO-V3     | https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/yolov3/ |
| PPYOLO      | https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/ppyolo/ |
| PPYOLO-Tiny | https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/ppyolo/ |
| PPYOLO-V2   | https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/ppyolo/ |
| TTFNet      | https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/ttfnet/ |
| PAFNet      | https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/ttfnet/ |
| SSD         | https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/ssd/ |
