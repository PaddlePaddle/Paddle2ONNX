# Paddle2ONNX

简体中文 | [English](README.md)

## 简介

paddle2onnx支持将**PaddlePaddle**模型格式转化到**ONNX**模型格式。通过ONNX可以完成将Paddle模型到多种推理引擎的部署，包括TensorRT/OpenVINO/MNN/TNN/NCNN，以及其它对ONNX开源格式进行支持的推理引擎或硬件。

## 环境依赖

- python >= 3.6
- paddlepaddle >= 2.1.0
- onnx >= 1.10.0

## 安装

```
pip install paddle2onnx
```

## 使用

### 获取PaddlePaddle部署模型

Paddle2ONNX在导出模型时，需要传入部署模型格式，包括两个文件  
- `model_name.pdmodel`: 表示模型结构  
- `model_name.pdiparams`: 表示模型参数
[注意] 这里需要注意，两个文件其中参数文件后辍为`.pdiparams`，如你的参数文件后辍是`.pdparams`，那说明你的参数是训练过程中保存的，当前还不是部署模型格式。 部署模型的导出可以参照各个模型套件的导出模型文档。


### 命令行转换

```
paddle2onnx --model_dir saved_inference_model \
            --model_filename model.pdmodel \
            --params_filename model.pdiparams \
            --save_file model.onnx
            --enable_dev_version True
```
如你有ONNX模型优化的需求，推荐使用`onnx-simplifier`，也可使用如下命令对模型进行优化
```
python -m paddle2onnx.optimize --input_model model.onnx --output_model new_model.onnx
```
如需要修改导出的模型输入形状，如改为静态shape
```
python -m paddle2onnx.optimize --input_model model.onnx \
                               --output_model new_model.onnx \
                               --input_shape_dict "{'x':[1,3,224,224]}"
```

#### 参数选项
| 参数 |参数说明 |
|----------|--------------|
|--model_dir | 配置包含Paddle模型的目录路径|
|--model_filename |**[可选]** 配置位于`--model_dir`下存储网络结构的文件名|
|--params_filename |**[可选]** 配置位于`--model_dir`下存储模型参数的文件名称|
|--save_file | 指定转换后的模型保存目录路径 |
|--opset_version | **[可选]** 配置转换为ONNX的OpSet版本，目前比较稳定地支持9、10、11三个版本，默认为9 |
|--enable_dev_version | **[可选]** 是否使用新版本Paddle2ONNX（当前正在开发中），默认为False |
|--enable_onnx_checker| **[可选]**  配置是否检查导出为ONNX模型的正确性, 建议打开此开关。若指定为True，需要安装 onnx>=1.7.0, 默认为False|
|--enable_auto_update_opset| **[可选]**  是否开启opset version自动升级,当低版本opset无法转换时，自动选择更高版本的opset 默认为True|
|--input_shape_dict| **[可选]**  配置输入的shape, 默认为空|
|--version |**[可选]** 查看paddle2onnx版本 |

- 使用onnxruntime验证转换模型, 请注意安装最新版本（最低要求1.10.0）：

## License
Provided under the [Apache-2.0 license](https://github.com/PaddlePaddle/paddle-onnx/blob/develop/LICENSE).
