# paddle2onnx
paddle2onnx支持将**PaddlePaddle**框架下产出的模型转化到**ONNX**模型格式.
paddle2onnx is a toolkit for converting trained model to **ONNX** from **PaddlePaddle** deep learning framework.

## 更新记录
2019.09.25
1、新增支持SE_ResNet50_vd、SqueezeNet1_0、SE_ResNext50_32x4d、Xception41、VGG16、InceptionV4、YoloV3模型转换
2、解决0.1版本无法适配新版ONNX版本问题
2109.08.20
1. 解决preview版本无法适配最新的PaddlePaddle和ONNX版本问题。
2. 功能上支持主流的图像分类模型和部分图像检测模型。
3. 统一开发者精度对齐测试框架，代码开发和贡献者可以通过内置的Operators单测和模型Layer测试框架来验证转换后的模型的精度。
4. 统一对外的使用接口，用户可利用PIP安装功能包进行使用。

## 环境依赖

### 1. 普通用户环境配置
     python >= 3.5  
     paddlepaddle >= 1.5.0
     onnx >= 1.5
### 2. 开发者环境配置
     python >= 3.5  
     paddlepaddle >= 1.5.0
     onnx >= 1.5
     torch >= 1.1
     onnxruntime >= 0.4.0
##  安装
###  安装方式1
     pip install paddle2onnx
### 安装方式2
     git clone https://github.com/PaddlePaddle/paddle-onnx.git
     python setup.py install
##  使用方式
###  普通用户使用方式
> 如果用户只是想将paddle模型转化成onnx模型，可以使用下面的命令进行操作。

    paddle2onnx --fluid_model src_dir  --onnx_model dist_name
###  开发者使用方式
> 如果用户有一个新的模型要转成onnx模型，想验证模型的精确度，可以使用下面的方式来进行验证。

    git clone https://github.com/PaddlePaddle/paddle-onnx.git
    python fluid_onnx/fluid_to_onnx.py --fluid_model src --onnx_model dist --debug
### 参数选项
| 参数 |参数说明 |
|----------|--------------|
|fluid_model | paddle fluid模型和模型参数所在目录 |
|onnx_model  | 转化成onnx模型的模型名称
|name_prefix| [可选]某些paddle模型的模型参数加了前缀，则需要指定模型参数前缀,例如@HUB_mobilenet_v2_imagenet@conv6_2_expand_bn_scale |
|fluid_model_name |[可选]如果导入的paddle模型不是默认__model__,需要指定模型的名字|
|fluid_params_name|[可选]如果导入的paddle模型参数是合并在一个文件里面，需要指定模型参数文件名|
|debug | [可选]如果开发者要对转化的模型进行精度测试，打开此开关 |
|return_variable| [可选]在debug模式中，如果paddle模型返回的结果是LoDTensor,需要打开此开关
|check_task| [可选]在debug模式中，根据不同配置项选择不同的执行器和数据构造器 |
|image_path | [可选]在debug模式中，可以选择加载不同的图片进行精度验证 |

##  相关文档
[paddle2onnx测试模型库](docs/model_zoo.md)

## 注意事项
1. 默认情况下，paddle2onnx工具是不提供Paddle模型进行转换的。PaddleHub提供了较多标准的模型供使用，用户可以拉取PaddleHub中的模型进行转化，安装PaddleHub的模型后会有提示模型安装位置，例如ssd模型安装位置在/root/paddle/paddle-onnx/ssd_mobilenet_v1_pascal，不同的PaddleHub的安装环境安装位置会有不同，用户请注意PaddleHub模型的安装位置。
2. 工具参数name_prefix的使用方式。使用paddle2onnx工具前最好观察一下Paddle模型的参数名字是否带有前缀，例如@HUB_mobilenet_v2_imagenet@conv6_2_expand_bn_scale，那么使用paddle2onnx需要加上参数 --name_prefix  @HUB_mobilenet_v2_imagenet@。默认情况下是不带前缀。
3. Model zoo的使用方式。Model zoo大部分是提供了PaddleHub模型的链接地址，用户可以通过安装PaddleHub模型来获取标准模型。目前PaddleHub没有集成densenet_121、InceptionV4、SE_ResNet50_vd、Xception41这四个模型，我们提供了PaddleCV库的下载地址，该模型不可以直接进行转化，用户需要使用save_inference_model接口来保存模型和参数。
4. 目前支持PaddleDetection中的YoloV3，我们提供了PaddleDetection中的yolov3_darknet_voc的下载地址，用户需要使用save_inference_model接口来保存模型和参数，并且yolov3中的multiclass_nms的normalized参数需手动设为True，否则最终结果会存在轻微的差异。
## License
Provided under the [Apache-2.0 license](https://github.com/PaddlePaddle/paddle-onnx/blob/develop/LICENSE).
