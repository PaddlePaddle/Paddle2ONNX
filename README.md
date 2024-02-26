# Paddle2ONNX

简体中文 | [English](README_en.md)

## 简介

Paddle2ONNX 支持将 **PaddlePaddle** 模型格式转化到 **ONNX** 模型格式。通过 ONNX 可以完成将 Paddle 模型到多种推理引擎的部署，包括 TensorRT/OpenVINO/MNN/TNN/NCNN，以及其它对 ONNX 开源格式进行支持的推理引擎或硬件。

感谢[EasyEdge团队](https://ai.baidu.com/easyedge/home)贡献的Paddle2Caffe, 支持将Paddle模型导出为Caffe格式，安装及使用方式参考[Paddle2Caffe](Paddle2Caffe)。


## 模型库
Paddle2ONNX 建设了一个飞桨热点模型的模型库，包括 PicoDet、OCR、HumanSeg 等多种领域模型，有需求的开发者可直接下载使用，进入目录[model_zoo](./model_zoo)了解更多详情！

## 环境依赖

- 无

## 安装

```
pip install paddle2onnx
```

- [Github 源码安装方式](docs/zh/compile.md)

## 使用

### 获取PaddlePaddle部署模型

Paddle2ONNX 在导出模型时，需要传入部署模型格式，包括两个文件  
- `model_name.pdmodel`: 表示模型结构  
- `model_name.pdiparams`: 表示模型参数
[注意] 这里需要注意，两个文件其中参数文件后辍为 `.pdiparams`，如你的参数文件后辍是 `.pdparams`，那说明你的参数是训练过程中保存的，当前还不是部署模型格式。 部署模型的导出可以参照各个模型套件的导出模型文档。


### 命令行转换

```
paddle2onnx --model_dir saved_inference_model \
            --model_filename model.pdmodel \
            --params_filename model.pdiparams \
            --save_file model.onnx \
            --enable_dev_version True
```
#### 参数选项
| 参数 |参数说明 |
|----------|--------------|
|--model_dir | 配置包含 Paddle 模型的目录路径|
|--model_filename |**[可选]** 配置位于 `--model_dir` 下存储网络结构的文件名|
|--params_filename |**[可选]** 配置位于 `--model_dir` 下存储模型参数的文件名称|
|--save_file | 指定转换后的模型保存目录路径 |
|--opset_version | **[可选]** 配置转换为 ONNX 的 OpSet 版本，目前支持 7~16 等多个版本，默认为 9 |
|--enable_onnx_checker| **[可选]**  配置是否检查导出为 ONNX 模型的正确性, 建议打开此开关， 默认为 False|
|--enable_auto_update_opset| **[可选]**  是否开启 opset version 自动升级功能，当低版本 opset 无法转换时，自动选择更高版本的 opset进行转换， 默认为 True|
|--deploy_backend |**[可选]** 量化模型部署的推理引擎，支持 onnxruntime、tensorrt 或 others，当选择 others 时，所有的量化信息存储于 max_range.txt 文件中，默认为 onnxruntime |
|--save_calibration_file |**[可选]** TensorRT 8.X版本部署量化模型需要读取的 cache 文件的保存路径，默认为 calibration.cache |
|--version |**[可选]** 查看 paddle2onnx 版本 |
|--external_filename |**[可选]** 当导出的 ONNX 模型大于 2G 时，需要设置 external data 的存储路径，推荐设置为：external_data |
|--export_fp16_model |**[可选]** 是否将导出的 ONNX 的模型转换为 FP16 格式，并用 ONNXRuntime-GPU 加速推理，默认为 False |
|--custom_ops |**[可选]** 将 Paddle OP 导出为 ONNX 的 Custom OP，例如：--custom_ops '{"paddle_op":"onnx_op"}，默认为 {} |

- 使用 onnxruntime 验证转换模型, 请注意安装最新版本（最低要求 1.10.0）


### 其他优化工具
1.  如你对导出的 ONNX 模型有优化的需求，推荐使用 `onnx-simplifier`，也可使用如下命令对模型进行优化
```
python -m paddle2onnx.optimize --input_model model.onnx --output_model new_model.onnx
```

2.  如需要修改导出 ONNX 的模型输入形状，如改为静态 shape
```
python -m paddle2onnx.optimize --input_model model.onnx \
                               --output_model new_model.onnx \
                               --input_shape_dict "{'x':[1,3,224,224]}"
```

3. 如果你有裁剪 Paddle 模型，固化或修改 Paddle 模型输入 Shape 或者合并 Paddle 模型的权重文件等需求，请使用如下工具：[Paddle 相关工具](./tools/paddle/README.md)

4. 如果你需要裁剪 ONNX 模型或者修改 ONNX 模型，请参考如下工具：[ONNX 相关工具](./tools/onnx/README.md)

5. PaddleSlim 量化模型导出请参考：[量化模型导出ONNX](./docs/zh/quantize.md)

## License
Provided under the [Apache-2.0 license](https://github.com/PaddlePaddle/paddle-onnx/blob/develop/LICENSE).
