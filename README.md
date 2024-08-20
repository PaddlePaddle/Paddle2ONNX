# Paddle2ONNX

简体中文 | [English](README_en.md)

# 1 Paddle2ONNX 简介

Paddle2ONNX 支持将 **PaddlePaddle** 模型格式转化到 **ONNX** 模型格式。通过 ONNX 可以完成将 Paddle 模型到多种推理引擎的部署，包括 TensorRT/OpenVINO/MNN/TNN/NCNN，以及其它对 ONNX 开源格式进行支持的推理引擎或硬件。

# 2 Paddle2ONNX 环境依赖

Paddle2ONNX 本身不依赖其他组件，但是我们建议您在以下环境下使用 Paddle2ONNX ：

- PaddlePaddle == 2.6.0
- onnxruntime >= 1.10.0

# 3 安装 Paddle2ONNX

如果您只是想要安装 Paddle2ONNX 且没有二次开发的需求，你可以通过执行以下代码来快速安装 Paddle2ONNX

```
pip install paddle2onnx
```

如果你希望对 Paddle2ONNX 进行二次开发，请按照[Github 源码安装方式](docs/zh/compile_local.md)编译Paddle2ONNX。

# 4 快速使用教程

## 4.1 获取PaddlePaddle部署模型

Paddle2ONNX 在导出模型时，需要传入部署模型格式，包括两个文件

- `model_name.pdmodel`: 表示模型结构
- `model_name.pdiparams`: 表示模型参数

## 4.2 调整Paddle模型

如果对Paddle模型的输入输出需要做调整，可以前往[Paddle 相关工具](./tools/paddle/README.md)查看教程。

## 4.3 使用命令行转换 PaddlePaddle 模型

你可以通过使用命令行并通过以下命令将Paddle模型转换为ONNX模型

```bash
paddle2onnx --model_dir saved_inference_model \
            --model_filename model.pdmodel \
            --params_filename model.pdiparams \
            --save_file model.onnx
```

可调整的转换参数如下表:

| 参数                         | 参数说明                                                                                                            |
|----------------------------|-----------------------------------------------------------------------------------------------------------------|
| --model_dir                | 配置包含 Paddle 模型的目录路径                                                                                             |
| --model_filename           | **[可选]** 配置位于 `--model_dir` 下存储网络结构的文件名                                                                         |
| --params_filename          | **[可选]** 配置位于 `--model_dir` 下存储模型参数的文件名称                                                                        |
| --save_file                | 指定转换后的模型保存目录路径                                                                                                  |
| --opset_version            | **[可选]** 配置转换为 ONNX 的 OpSet 版本，目前支持 7~16 等多个版本，默认为 9                                                            |
| --enable_onnx_checker      | **[可选]**  配置是否检查导出为 ONNX 模型的正确性, 建议打开此开关， 默认为 False                                                             |
| --enable_auto_update_opset | **[可选]**  是否开启 opset version 自动升级功能，当低版本 opset 无法转换时，自动选择更高版本的 opset进行转换， 默认为 True                              |
| --deploy_backend           | **[可选]** 量化模型部署的推理引擎，支持 onnxruntime、tensorrt 或 others，当选择 others 时，所有的量化信息存储于 max_range.txt 文件中，默认为 onnxruntime |
| --save_calibration_file    | **[可选]** TensorRT 8.X版本部署量化模型需要读取的 cache 文件的保存路径，默认为 calibration.cache                                          |
| --version                  | **[可选]** 查看 paddle2onnx 版本                                                                                      |
| --external_filename        | **[可选]** 当导出的 ONNX 模型大于 2G 时，需要设置 external data 的存储路径，推荐设置为：external_data                                       |
| --export_fp16_model        | **[可选]** 是否将导出的 ONNX 的模型转换为 FP16 格式，并用 ONNXRuntime-GPU 加速推理，默认为 False                                           |


## 4.4 裁剪ONNX

如果你需要调整 ONNX 模型，请参考 [ONNX 相关工具](./tools/onnx/README.md)

## 4.5 优化ONNX

如你对导出的 ONNX 模型有优化的需求，推荐使用 `onnx-simplifier`，也可使用如下命令对模型进行优化

```
pip install onnxslim
onnxslim model.onnx slim.onnx
```

# 5 代码贡献

繁荣的生态需要大家的携手共建，开发者可以参考 [Paddle2ONNX 贡献指南](./docs/zh/Paddle2ONNX_Development_Guide.md) 来为 Paddle2ONNX 贡献代码。

# 6 License

Provided under the [Apache-2.0 license](https://github.com/PaddlePaddle/paddle-onnx/blob/develop/LICENSE).

# 7 感谢捐赠

* 感谢 PaddlePaddle 团队提供服务器支持 Paddle2ONNX 的 CI 建设。
* 感谢社区用户 [chenwhql](https://github.com/chenwhql), [luotao1](https://github.com/luotao1), [goocody](https://github.com/goocody), [jeff41404](https://github.com/jeff41404), [jzhang553](https://github.com/jzhang533), [ZhengBicheng](https://github.com/ZhengBicheng) 于2024年03月28日向 Paddle2ONNX PMC 捐赠共 10000 元人名币用于 Paddle2ONNX 的发展。
