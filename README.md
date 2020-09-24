## Paddle2ONNX

paddle2onnx支持将**PaddlePaddle**框架下产出的模型转化到**ONNX**模型格式.
paddle2onnx is a toolkit for converting trained model to **ONNX** from **PaddlePaddle** deep learning framework.

## 更新记录

2020.9.21
1. 支持ONNX Opset 9, 10, 11三个版本的导出。
2. 新增支持转换的OP: swish ,floor, uniform_random, abs, instance_norm, clip, tanh, log和pad2d。

2019.09.25
1. 新增支持SE_ResNet50_vd、SqueezeNet1_0、SE_ResNext50_32x4d、Xception41、VGG16、InceptionV4、YoloV3模型转换。
2. 解决0.1版本无法适配新版ONNX版本问题。

2109.08.20
1. 解决preview版本无法适配最新的PaddlePaddle和ONNX版本问题。
2. 功能上支持主流的图像分类模型和部分图像检测模型。
3. 统一开发者精度对齐测试框架，代码开发和贡献者可以通过内置的Operators单测和模型Layer测试框架来验证转换后的模型的精度。
4. 统一对外的使用接口，用户可利用PIP安装功能包进行使用。

## 环境依赖

### 用户环境配置

     python >= 3.5  
     paddlepaddle >= 1.6.0
     onnx >= 1.6

##  安装
###  安装方式1

     pip install paddle2onnx

### 安装方式2

     git clone https://github.com/PaddlePaddle/paddle2onnx.git
     python setup.py install

##  使用方式
###  普通用户使用方式
> 如果用户只是想将paddle模型转化成onnx模型，可以使用下面的命令进行操作。

    paddle2onnx --model src_dir  --save_dir dist_dir

### 参数选项
| 参数 |参数说明 |
|----------|--------------|
|--model | 指定包含Paddle模型和参数：'__model__', '__params__'的路径 |
|--save_dir | 指定转换后的模型保存目录路径 |
|--onnx_opset | **[可选]** 该参数可设置转换为ONNX的OpSet版本，目前支持9、10、11，默认为10 |


##  相关文档
[paddle2onnx测试模型库](docs/model_zoo.md)

## 注意事项
1. 默认情况下，paddle2onnx工具是不提供Paddle模型进行转换的。PaddleHub提供了较多标准的模型供使用，用户可以拉取PaddleHub中的模型进行转化，安装PaddleHub的模型后会有提示模型安装位置，例如ssd模型安装位置在/root/paddle/paddle-onnx/ssd_mobilenet_v1_pascal，不同的PaddleHub的安装环境安装位置会有不同，用户请注意PaddleHub模型的安装位置。
2. 工具参数name_prefix的使用方式。使用paddle2onnx工具前最好观察一下Paddle模型的参数名字是否带有前缀，例如@HUB_mobilenet_v2_imagenet@conv6_2_expand_bn_scale，那么使用paddle2onnx需要加上参数 --name_prefix  @HUB_mobilenet_v2_imagenet@。默认情况下是不带前缀。
3. Model zoo的使用方式。Model zoo大部分是提供了PaddleHub模型的链接地址，用户可以通过安装PaddleHub模型来获取标准模型。目前PaddleHub没有集成densenet_121、InceptionV4、SE_ResNet50_vd、Xception41这四个模型，我们提供了PaddleCV库的下载地址，该模型不可以直接进行转化，用户需要使用save_inference_model接口来保存模型和参数。

PaddleX, PaddleClas, PaddleSeg和PaddleOCR目前支持模型如下:

- 支持[PaddleX](https://github.com/PaddlePaddle/PaddleX)和[PaddleClas](https://github.com/PaddlePaddle/PaddleCLAS)中的所有分类模型
- 支持[PaddleX](https://github.com/PaddlePaddle/PaddleX)和[PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg)中的UNet/DeepLabV3/HRNet语义分割模型
- 支持[PaddleX](https://github.com/PaddlePaddle/PaddleX)中YOLOv3的检测模型
- 支持[PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)中的文字检测模型（文字识别模型暂不支持)

## License
Provided under the [Apache-2.0 license](https://github.com/PaddlePaddle/paddle-onnx/blob/develop/LICENSE).
