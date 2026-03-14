# Paddle2ONNX

[简体中文](README.md) | English

## 1 Introduction

Paddle2ONNX supports the conversion of PaddlePaddle model format to ONNX model format. Through ONNX, it is possible to deploy Paddle models to various inference engines, including TensorRT/OpenVINO/MNN/TNN/NCNN, as well as other inference engines or hardware that support the ONNX open-source format.

## 2 Paddle2ONNX Environment Dependencies

Paddle2ONNX depends on PaddlePaddle 3.3+, we recommend using Paddle2ONNX in the following environments:

- Python >= 3.10
- PaddlePaddle >= 3.3
- ONNX >= 1.20.1
- protobuf >= 34.0
- onnxruntime >= 1.10.0

## 3 Install Paddle2ONNX

If you only want to install Paddle2ONNX without the need for secondary development, you can quickly install Paddle2ONNX by executing the following code.

```
pip install paddle2onnx
```

If you wish to conduct secondary development on Paddle2ONNX, please follow the [GitHub Source Code Installation Method](docs/en/compile_local.md) to compile Paddle2ONNX.

## 4 Quick Start Tutorial

### 4.1 Get the PaddlePaddle Deployment Model

When Paddle2ONNX exports the model, it needs to pass in the deployment model format, including two files
- `model_name.json`: Indicates the model structure
- `model_name.pdiparams`: Indicates model parameters

### 4.2 Adjusting Paddle Models

If adjustments to the input and output of Paddle models are needed, you can visit the [Paddle related tools](./tools/paddle/README.md) for tutorials.

### 4.3 Using Command Line to Convert PaddlePaddle Models

You can convert Paddle models to ONNX models using the command line with the following command:

```bash
paddle2onnx --model_dir model_dir \
            --model_filename model.json \
            --params_filename model.pdiparams \
            --save_file model.onnx
```

The adjustable conversion parameters are listed in the following table:

| Parameter                  | Parameter Description                                                                                                                                                                                                             |
|----------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| --model_dir                | Configure directory path containing Paddle models                                                                                                                                                                                 |
| --model_filename           | **[Optional]** Configure the filename to store the network structure under `--model_dir`                                                                                                                                          |
| --params_filename          | **[Optional]** Configure the name of the file to store model parameters under `--model_dir`                                                                                                                                       |
| --save_file                | Specify the converted model save directory path                                                                                                                                                                                   |
| --opset_version            | **[Optional]** Configure the OpSet version converted to ONNX, currently supports multiple versions such as 7~19, the default is 9                                                                                                 |
| --enable_auto_update_opset | **[Optional]** Whether to enable the opset version automatic upgrade function, when the lower version of the opset cannot be converted, automatically select the higher version of the opset for conversion, the default is True  |
| --enable_onnx_checker      | **[Optional]** Configure whether to check the correctness of the exported ONNX model, it is recommended to turn on this switch, the default is True                                                                              |
| --enable_dist_prim_all     | **[Optional]** Whether to enable the decomposition of combined operators, the default is False
| --optimize_tool            | **[Optional]** ONNX model optimization tool, options: onnxoptimizer, polygraphy, or None, the default is onnxoptimizer.
| --enable_verbose           | **[Optional]** Whether to show verbose logs, default False |
| --version                  | **[Optional]** View paddle2onnx version       |
### 4.4 Pruning ONNX

If you need to adjust ONNX models, please refer to [ONNX related tools](./tools/onnx/README.md)

### 4.5 Optimize ONNX

If you have optimization needs for the exported ONNX model, we recommend using `onnx-simplifier`. You can also optimize the model using the following command.

```
pip install onnxslim
onnxslim model.onnx slim.onnx
```

## 5 Code Contribution

A thriving ecosystem requires everyone's collaborative efforts. Developers can refer to the [Paddle2ONNX Contribution Guide](./docs/zh/Paddle2ONNX_Development_Guide.md) to contribute code to Paddle2ONNX.

## 6 License

Provided under the [Apache-2.0 license](https://github.com/PaddlePaddle/paddle-onnx/blob/develop/LICENSE).

## 7 Thank you for the donation

* Thanks to the PaddlePaddle team for providing server support for the CI infrastructure of Paddle2ONNX.
* Thanks to community users [chenwhql](https://github.com/chenwhql), [luotao1](https://github.com/luotao1), [goocody](https://github.com/goocody), [jeff41404](https://github.com/jeff41404), [jzhang553](https://github.com/jzhang533), [ZhengBicheng](https://github.com/ZhengBicheng) for donating a total of 10,000 RMB to the Paddle2ONNX PMC on March 28, 2024, for the development of Paddle2ONNX.
