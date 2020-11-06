## Paddle2ONNX

paddle2onnx支持将**PaddlePaddle**框架下产出的模型转化到**ONNX**模型格式.
paddle2onnx is a toolkit for converting trained model to **ONNX** from **PaddlePaddle** deep learning framework.

## 更新记录

2020.11.4
1. 支持Paddle动态图模型导出为ONNX。
2. 重构代码结构以更好地支持不同Paddle版本，以及动态图和静态图的转换。

2020.9.21
1. 支持ONNX Opset 9, 10, 11三个版本的导出。
2. 新增支持转换的OP: swish, floor, uniform_random, abs, instance_norm, clip, tanh, log, norm和pad2d。

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

     python >= 2.7  
     静态图: paddlepaddle >= 1.8.0
     动态图: paddlepaddle >= 2.0.0
     onnx == 1.7.0 | 可选

##  安装
###  安装方式1

     pip install paddle2onnx

### 安装方式2

     git clone https://github.com/PaddlePaddle/paddle2onnx.git
     python setup.py install

##  使用方式
### 静态图模型导出

#### 命令行

    paddle2onnx --model_dir paddle_model  --save_file onnx_file --opset_version 10 --enable_onnx_checker True

#### 参数选项
| 参数 |参数说明 |
|----------|--------------|
|--model_dir | 指定包含Paddle模型, 由`paddle.fluid.io.save_inference_model`保存得到|
|--model_filename |**[可选]** 用于指定位于`--model_dir`下存储网络结构的文件名称。当且仅当所有模型参数被保存在一个单独的二进制文件中，它才需要被指定。默认为None|
|--params_filename |**[可选]** 用于指定位于`--model_dir`下存储所有模型参数的文件名称。当且仅当所有模型参数被保存在一个单独的二进制文件中，它才需要被指定。默认为None|
|--save_file | 指定转换后的模型保存目录路径 |
|--opset_version | **[可选]** 该参数可设置转换为ONNX的OpSet版本，目前比较稳定地支持9、10、11三个版本，默认为9 |
|--enable_onnx_checker| **[可选]**  是否检查导出为ONNX模型的正确性, 建议打开此开关。若指定为True，需要安装 pip install onnx==1.7.0, 默认为False|
|--version |**[可选]** 查看paddle2onnx版本 |

> 补充说明：
>
> - PaddlePaddle模型的两种存储形式：
>    - 参数被保存在一个单独的二进制文件中（combined），需要在指定--model_dir的前提下，指定--model_filename, --params_filename, 分别表示--model_dir目录下的网络文件名称和参数文件名称。
>    - 参数被保存为多个文件（not combined），只需要指定--model_dir，该目录下面需要包含了'\_\_model\_\_'，以及多个参数文件。

#### 简单教程
- [静态图导出ONNX教程](docs/tutorial.ipynb)

### 动态图模型导出

处于实验状态，Paddle 2.0正式版发布后，会提供详细使用教程。

##  相关文档

- [paddle2onnx测试模型库](docs/model_zoo.md)
- [paddle2onnx支持转换算子列表](docs/op_list.md)


## License
Provided under the [Apache-2.0 license](https://github.com/PaddlePaddle/paddle-onnx/blob/develop/LICENSE).
