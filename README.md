# Paddle2ONNX

简体中文 | [English](README_en.md)

# 1 简介

Paddle2ONNX 支持将 **PaddlePaddle** 模型格式转化到 **ONNX** 模型格式。通过 ONNX 可以完成将 Paddle 模型到多种推理引擎的部署，包括
TensorRT/OpenVINO/MNN/TNN/NCNN，以及其它对 ONNX 开源格式进行支持的推理引擎或硬件。

感谢[EasyEdge团队](https://ai.baidu.com/easyedge/home)贡献的Paddle2Caffe,
支持将Paddle模型导出为Caffe格式，安装及使用方式参考[Paddle2Caffe](Paddle2Caffe)。

# 2 环境依赖

- PaddlePaddle 2.6.0
- onnxruntime >= 1.10.0

# 3 安装

针对PaddlePaddle2.5.2的用户可以直接运行以下命令行代码来安装P2O

```
pip install paddle2onnx
```

由于没有自动发包机制，针对PaddlePaddle2.6.0的用户，请按照[Github 源码安装方式](docs/zh/compile.md)编译Paddle2ONNX。

# 4 使用

## 4.1 获取PaddlePaddle部署模型

Paddle2ONNX 在导出模型时，需要传入部署模型格式，包括两个文件

- `model_name.pdmodel`: 表示模型结构
- `model_name.pdiparams`: 表示模型参数
  [注意] 这里需要注意，两个文件其中参数文件后辍为 `.pdiparams`，如你的参数文件后辍是 `.pdparams`
  ，那说明你的参数是训练过程中保存的，当前还不是部署模型格式。 部署模型的导出可以参照各个模型套件的导出模型文档。

## 4.2 调整Paddle模型

如果对Paddle模型的输入输出需要做调整，可以前往[Paddle 相关工具](./tools/paddle/README.md)查看教程。

## 4.3 命令行转换

使用如下命令将Paddle模型转换为ONNX模型

```
paddle2onnx --model_dir saved_inference_model \
            --model_filename model.pdmodel \
            --params_filename model.pdiparams \
            --save_file model.onnx \
            --enable_dev_version True
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
| --custom_ops               | **[可选]** 将 Paddle OP 导出为 ONNX 的 Custom OP，例如：--custom_ops '{"paddle_op":"onnx_op"}，默认为 {}                       |


## 4.4 裁剪ONNX

如果你需要调整 ONNX 模型，请参考如下工具：[ONNX 相关工具](./tools/onnx/README.md)

## 4.5 优化ONNX

如你对导出的 ONNX 模型有优化的需求，推荐使用 `onnx-simplifier`，也可使用如下命令对模型进行优化

```
python -m paddle2onnx.optimize --input_model model.onnx --output_model new_model.onnx
```

# 5 License

Provided under the [Apache-2.0 license](https://github.com/PaddlePaddle/paddle-onnx/blob/develop/LICENSE).

# 6 捐赠

* 感谢PaddlePaddle团队提供服务器支持Paddle2ONNX的CI建设
* 感谢社区用户 [chenwhql](https://github.com/chenwhql)、[luotao1](https://github.com/luotao1)、
  [goocody](https://github.com/goocody)、[jeff41404](https://github.com/jeff41404)、
  [jzhang553](https://github.com/jzhang533)、[ZhengBicheng](https://github.com/ZhengBicheng)
  与2024年03月28日向Paddle2ONNX PMC捐赠共10000元人名币用于Paddle2ONNX的发展。