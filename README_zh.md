# Paddle2ONNX

简体中文 | [English](README.md)

## 简介

paddle2onnx支持将**PaddlePaddle**模型格式转化到**ONNX**模型格式。

- 模型格式，支持Paddle静态图和动态图模型转为ONNX，可转换由[save_inference_model](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/static/save_inference_model_cn.html#save-inference-model)导出的静态图模型，使用方法请参考[IPthon示例](examples/tutorial.ipynb)。动态图转换目前处于实验状态，将伴随Paddle 2.0正式版发布后，提供详细使用教程。
- 算子支持，目前稳定支持导出ONNX Opset 9~11，部分Paddle算子支持更低的ONNX Opset转换，详情可参考[算子列表](docs/zh/op_list.md)。
- 模型类型，官方测试可转换的模型请参考[模型库](docs/zh/model_zoo.md)。

## AIStudio入门教程

- [Paddle2.0导出ONNX模型和推理](https://aistudio.baidu.com/aistudio/projectdetail/1461212)
- [手把手教你使用ONNXRunTime部署PP-OCR](https://aistudio.baidu.com/aistudio/projectdetail/1479970)

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

Paddle模型的参数保存为多个文件（not combined）:

    paddle2onnx --model_dir paddle_model  --save_file onnx_file --opset_version 10 --enable_onnx_checker True

Paddle模型的参数保存在一个单独的二进制文件中（combined）:

    paddle2onnx --model_dir paddle_model  --model_filename model_filename --params_filename params_filename --save_file onnx_file --opset_version 10 --enable_onnx_checker True

#### 参数选项
| 参数 |参数说明 |
|----------|--------------|
|--model_dir | 配置包含Paddle模型的路径, 由`paddle.fluid.io.save_inference_model`保存得到|
|--model_filename |**[可选]** 配置位于`--model_dir`下存储网络结构的文件名称。当且仅当所有模型参数被保存在一个单独的二进制文件中，它才需要被指定。默认为None|
|--params_filename |**[可选]** 配置位于`--model_dir`下存储模型参数的文件名称。当且仅当所有模型参数被保存在一个单独的二进制文件中，它才需要被指定。默认为None|
|--save_file | 指定转换后的模型保存目录路径 |
|--opset_version | **[可选]** 配置转换为ONNX的OpSet版本，目前比较稳定地支持9、10、11三个版本，默认为9 |
|--enable_onnx_checker| **[可选]**  配置是否检查导出为ONNX模型的正确性, 建议打开此开关。若指定为True，需要安装 onnx>=1.7.0, 默认为False|
|--version |**[可选]** 查看paddle2onnx版本 |

- PaddlePaddle模型的两种存储形式：
   - 参数被保存在一个单独的二进制文件中（combined），需要在指定--model_dir的前提下，指定--model_filename, --params_filename, 分别表示--model_dir目录下的网络文件名称和参数文件名称。
   - 参数被保存为多个文件（not combined），只需要指定--model_dir，该目录下面需要包含了'\_\_model\_\_'，以及多个参数文件。

#### IPython教程

- [静态图导出ONNX教程](examples/tutorial.ipynb)

### 动态图模型导出

```
import paddle
from paddle import nn
from paddle.static import InputSpec
import paddle2onnx as p2o

class LinearNet(nn.Layer):
    def __init__(self):
        super(LinearNet, self).__init__()
        self._linear = nn.Linear(784, 10)

    def forward(self, x):
        return self._linear(x)

layer = LinearNet()

# configure model inputs
x_spec = InputSpec([None, 784], 'float32', 'x')

# convert model to inference mode
layer.eval()

save_path = 'onnx.save/linear_net'
p2o.dygraph2onnx(layer, save_path + '.onnx', input_spec=[x_spec])

# when you paddlepaddle>2.0.0, you can try:
# paddle.onnx.export(layer, save_path, input_spec=[x_spec])

```

#### IPython教程

- [动态图导出ONNX教程](examples/tutorial_dygraph2onnx.ipynb)

##  相关文档

- [模型库](docs/zh/model_zoo.md)
- [算子列表](docs/zh/op_list.md)
- [更新记录](docs/zh/change_log.md)

## License
Provided under the [Apache-2.0 license](https://github.com/PaddlePaddle/paddle-onnx/blob/develop/LICENSE).
