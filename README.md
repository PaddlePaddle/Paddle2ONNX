## Paddle2ONNX

paddle2onnx支持将**PaddlePaddle**框架下产出的模型转化到**ONNX**模型格式.
paddle2onnx is a toolkit for converting trained model to **ONNX** from **PaddlePaddle** deep learning framework.

## 更新记录

2020.11.5
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

    paddle2onnx --model_dir paddle_model  --save_file onnx_file --onnx_opset 10 --enable_onnx_checker True

### 动态图模型导出

```
import paddle
import numpy as np

class Model(paddle.nn.Layer):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y, z=False):
        if z:
            return x + y, x - y
        else:
            return x * y, x / y

def export_with_input_spec():
    paddle.enable_dygraph()
    model = Model()
    x_spec = paddle.static.InputSpec(shape=[None, 4], dtype='float32', name='x')
    y_spec = paddle.static.InputSpec(shape=[None, 4], dtype='float32', name='y')
    paddle.onnx.export(model, 'dynamic_input.onnx', input_spec=[x_spec, y_spec])

def export_with_input_variable():
    paddle.enable_dygraph()
    model = Model()
    x =  paddle.to_tensor(np.array([1]).astype('float32'), name='x')
    y =  paddle.to_tensor(np.array([1]).astype('float32'), name='y')
    model = paddle.jit.to_static(model)
    out = model(x, y, z=True)
    paddle.onnx.export(model, 'pruned.onnx', input_spec=[x, y], output_spec=[out[0]])

#export model with InputSpec, which supports set dynamic shape for input.
export_with_input_spec()

#export model with Variable, which supports prune model by set 'output_spec' with output of model.
export_with_input_variable()
```  

### 参数选项
| 参数 |参数说明 |
|----------|--------------|
|--model_dir | 指定包含Paddle模型'\_\_model\_\_'和参数'\_\_params\_\_'的路径 |
|--save_file | 指定转换后的模型保存目录路径 |
|--onnx_opset | **[可选]** 该参数可设置转换为ONNX的OpSet版本，目前比较稳定地支持9、10、11三个版本，默认为10 |
|--enable_onnx_checker| **[可选]**  是否检查导出为ONNX模型的正确性, 建议打开此开关。若指定为True，需要安装 pip install onnx==1.7.0, 默认为False|
|--version |**[可选]** 查看paddle2onnx版本 |

##  相关文档
[paddle2onnx测试模型库](docs/model_zoo.md)
[paddle2onnx支持准换算子列表](docs/op_list.md)

## 注意事项
1. 默认情况下，paddle2onnx工具是不提供Paddle模型进行转换的。PaddleHub提供了较多标准的模型供使用，用户可以拉取PaddleHub中的模型进行转化，安装PaddleHub的模型后会有提示模型安装位置，例如ssd模型安装位置在/root/paddle/paddle-onnx/ssd_mobilenet_v1_pascal，不同的PaddleHub的安装环境安装位置会有不同，用户请注意PaddleHub模型的安装位置。
2. 工具参数name_prefix的使用方式。使用paddle2onnx工具前最好观察一下Paddle模型的参数名字是否带有前缀，例如@HUB_mobilenet_v2_imagenet@conv6_2_expand_bn_scale，那么使用paddle2onnx需要加上参数 --name_prefix  @HUB_mobilenet_v2_imagenet@。默认情况下是不带前缀。
3. Model zoo的使用方式。Model zoo大部分是提供了PaddleHub模型的链接地址，用户可以通过安装PaddleHub模型来获取标准模型。目前PaddleHub没有集成densenet_121、InceptionV4、SE_ResNet50_vd、Xception41这四个模型，我们提供了PaddleCV库的下载地址，该模型不可以直接进行转化，用户需要使用save_inference_model接口来保存模型和参数。

## License
Provided under the [Apache-2.0 license](https://github.com/PaddlePaddle/paddle-onnx/blob/develop/LICENSE).
