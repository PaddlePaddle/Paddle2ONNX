# Paddle2ONNX

简体中文 | [English](README_en.md)

## 1 Paddle2ONNX 简介

Paddle2ONNX 支持将 **PaddlePaddle** 模型格式转化到 **ONNX** 模型格式。通过 ONNX 可以完成将 Paddle 模型到多种推理引擎的部署，包括 TensorRT/OpenVINO/MNN/TNN/NCNN，以及其它对 ONNX 开源格式进行支持的推理引擎或硬件。

## 2 Paddle2ONNX 环境依赖

Paddle2ONNX 依赖PaddlePaddle3.0，我们建议您在以下环境下使用 Paddle2ONNX ：

- PaddlePaddle == 3.0.0
- onnxruntime >= 1.10.0

## 3 安装 Paddle2ONNX

如果您只是想要安装 Paddle2ONNX 且没有二次开发的需求，你可以通过执行以下代码来快速安装 Paddle2ONNX

```
pip install paddle2onnx
```

如果你希望对 Paddle2ONNX 进行二次开发，请按照[Github 源码安装方式](docs/zh/compile_local.md)编译Paddle2ONNX。

## 4 快速使用教程

### 4.1 获取PaddlePaddle部署模型

Paddle2ONNX 在导出模型时，需要传入部署模型格式，包括两个文件

- `model_name.json`: 表示模型结构
- `model_name.pdiparams`: 表示模型参数

### 4.2 调整Paddle模型

如果对Paddle模型的输入输出需要做调整，可以前往[Paddle 相关工具](./tools/paddle/README.md)查看教程。

### 4.3 使用命令行转换 PaddlePaddle 模型

你可以通过使用命令行并通过以下命令将Paddle模型转换为ONNX模型

```bash
paddle2onnx --model_dir model_dir \
            --model_filename model.json \
            --params_filename model.pdiparams \
            --save_file model.onnx
```

可调整的转换参数如下表:

| 参数                         | 参数说明                                                                                                            |
|----------------------------|-----------------------------------------------------------------------------------------------------------------|
| --model_dir                | 配置包含 Paddle 模型的目录路径                                                                                      |
| --model_filename           | **[可选]** 配置位于 `--model_dir` 下存储网络结构的文件名                                                              |
| --params_filename          | **[可选]** 配置位于 `--model_dir` 下存储模型参数的文件名                                                              |
| --save_file                | 指定转换后的模型保存目录路径                                                                                         |
| --opset_version            | **[可选]** 配置转换为ONNX的OpSet版本，目前支持7~19等多个版本，默认为 9                                                  |
| --enable_auto_update_opset | **[可选]** 是否开启opset version自动升级功能，当低版本opset无法转换时，自动选择更高版本的opset进行转换， 默认为 True          |
| --enable_onnx_checker      | **[可选]** 配置是否检查导出为 ONNX 模型的正确性, 建议打开此开关， 默认为 True                                             |
| --enable_dist_prim_all     | **[可选]** 是否开启组合算子拆解，默为 False                                                                           |
| --enable_optimization      | **[可选]** 是否开启模型优化，默认为 True                                                                             |
| --enable_verbose           | **[可选]** 是否打印更更详细的日志信息，默认为 False                                                                    |
| --version                  | **[可选]** 查看 paddle2onnx 版本                                                                                   |


### 4.4 裁剪ONNX

如果你需要调整 ONNX 模型，请参考 [ONNX 相关工具](./tools/onnx/README.md)

### 4.5 优化ONNX

如你对导出的 ONNX 模型有优化的需求，推荐使用 `onnxslim` 对模型进行优化:

```bash
pip install onnxslim
onnxslim model.onnx slim.onnx
```

## 5 代码贡献

繁荣的生态需要大家的携手共建，开发者可以参考 [Paddle2ONNX 贡献指南](./docs/zh/Paddle2ONNX_Development_Guide.md) 来为 Paddle2ONNX 贡献代码。

## 6 License

Provided under the [Apache-2.0 license](https://github.com/PaddlePaddle/paddle-onnx/blob/develop/LICENSE).

## 7 感谢捐赠

* 感谢 PaddlePaddle 团队提供服务器支持 Paddle2ONNX 的 CI 建设。
* 感谢社区用户 [chenwhql](https://github.com/chenwhql), [luotao1](https://github.com/luotao1), [goocody](https://github.com/goocody), [jeff41404](https://github.com/jeff41404), [jzhang553](https://github.com/jzhang533), [ZhengBicheng](https://github.com/ZhengBicheng) 于2024年03月28日向 Paddle2ONNX PMC 捐赠共 10000 元人名币用于 Paddle2ONNX 的发展。
