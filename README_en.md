# Paddle2ONNX

[简体中文](README.md) | English

## 🆕 New open source project FastDeploy
If the purpose of your conversion is to deploy TensorRT, OpenVINO, ONNX Runtime, the current PaddlePaddle provides [FastDeploy] (https://github.com/PaddlePaddle/FastDeploy), which supports 150+ models to be directly deployed to these engines, Paddle2ONNX The conversion process also no longer needs to be explicitly called by the user, helping everyone to solve various tricks and alignment problems during the conversion process.

- Welcome Star🌟 [https://github.com/PaddlePaddle/FastDeploy](https://github.com/PaddlePaddle/FastDeploy)
- [Use ONNX Runtime to deploy Paddle model C++ & Python](https://github.com/PaddlePaddle/FastDeploy/tree/develop/examples/runtime)
- [Use OpenVINO to deploy Paddle model C++ & Python](https://github.com/PaddlePaddle/FastDeploy/tree/develop/examples/runtime)
- [Use TensorRT to deploy Paddle model C++ & Python](https://github.com/PaddlePaddle/FastDeploy/tree/develop/examples/runtime)
- [PaddleOCR Model Deployment C++ & Python](https://github.com/PaddlePaddle/FastDeploy/tree/develop/examples/vision/ocr)
- [PaddleDetection Model Deployment C++ & Python](https://github.com/PaddlePaddle/FastDeploy/tree/develop/examples/vision/detection/paddledetection)

## Introduction

Paddle2ONNX supports converting **PaddlePaddle** model format to **ONNX** model format. The deployment of the Paddle model to a variety of inference engines can be completed through ONNX, including TensorRT/OpenVINO/MNN/TNN/NCNN, and other inference engines or hardware that support the ONNX open source format.

Thanks to [EasyEdge Team](https://ai.baidu.com/easyedge/home) for contributing Paddle2Caffe, which supports exporting the Paddle model to Caffe format. For installation and usage, please refer to [Paddle2Caffe](Paddle2Caffe).

## Model Zoo
Paddle2ONNX has built a model zoo of paddle popular models, including PicoDet, OCR, HumanSeg and other domain models. Developers who need it can directly download and use them. Enter the directory [model_zoo](./model_zoo) for more details!

## Environment dependencies

- none

## Install

```
pip install paddle2onnx
```

- [Github source installation method](docs/zh/compile.md)

## use

### Get the PaddlePaddle deployment model

When Paddle2ONNX exports the model, it needs to pass in the deployment model format, including two files
- `model_name.pdmodel`: Indicates the model structure
- `model_name.pdiparams`: Indicates model parameters
[Note] It should be noted here that the suffix of the parameter file in the two files is `.pdiparams`. If the suffix of your parameter file is `.pdparams`, it means that your parameters are saved during the training process and are not currently deployed. model format. The export of the deployment model can refer to the export model document of each model suite.


### Command line conversion

```
paddle2onnx --model_dir saved_inference_model \
             --model_filename model.pdmodel \
             --params_filename model.pdiparams\
             --save_file model.onnx \
             --enable_dev_version True
```
#### Parameter options
| Parameter |Parameter Description |
|----------|--------------|
|--model_dir | Configure directory path containing Paddle models|
|--model_filename |**[Optional]** Configure the filename to store the network structure under `--model_dir`|
|--params_filename |**[Optional]** Configure the name of the file to store model parameters under `--model_dir`|
|--save_file | Specify the converted model save directory path |
|--opset_version | **[Optional]** Configure the OpSet version converted to ONNX, currently supports multiple versions such as 7~16, the default is 9 |
|--enable_dev_version | **[Optional]** Whether to use the new version of Paddle2ONNX (recommended), the default is True |
|--enable_onnx_checker| **[Optional]** Configure whether to check the correctness of the exported ONNX model, it is recommended to turn on this switch, the default is False|
|--enable_auto_update_opset| **[Optional]** Whether to enable the opset version automatic upgrade function, when the lower version of the opset cannot be converted, automatically select the higher version of the opset for conversion, the default is True|
|--deploy_backend |**[Optional]** Inference engine for quantitative model deployment, supports onnxruntime, tensorrt or others, when other is selected, all quantization information is stored in the max_range.txt file, the default is onnxruntime |
|--save_calibration_file |**[Optional]** TensorRT 8.X version deploys the cache file that needs to be read to save the path of the quantitative model, the default is calibration.cache |
|--version |**[Optional]** View paddle2onnx version |
|--external_filename |**[Optional]** When the exported ONNX model is larger than 2G, you need to set the storage path of external data, the recommended setting is: external_data |
|--export_fp16_model |**[Optional]** Whether to convert the exported ONNX model to FP16 format, and use ONNXRuntime-GPU to accelerate inference, the default is False |
|--custom_ops |**[Optional]** Export Paddle OP as ONNX's Custom OP, for example: --custom_ops '{"paddle_op":"onnx_op"}, default is {} |

- Use ONNXRuntime to validate converted models, please pay attention to install the latest version (minimum requirement 1.10.0)

### Other optimization tools
1. If you need to optimize the exported ONNX model, it is recommended to use `onnx-simplifier`, or you can use the following command to optimize the model
```
python -m paddle2onnx.optimize --input_model model.onnx --output_model new_model.onnx
```

2. If you need to modify the input shape of the model exported to ONNX, such as changing to a static shape
```
python -m paddle2onnx.optimize --input_model model.onnx \
                                --output_model new_model.onnx \
                                --input_shape_dict "{'x':[1,3,224,224]}"
```

3. If you need to crop the Paddle model, solidify or modify the input Shape of the Paddle model, or merge the weight files of the Paddle model, please use the following tools: [Paddle-related tools](./tools/paddle/README.md)

4. If you need to crop the ONNX model or modify the ONNX model, please refer to the following tools: [ONNX related tools](./tools/onnx/README.md)

5. For PaddleSlim quantization model export, please refer to: [Quantization Model Export ONNX](./docs/zh/quantize.md)

### Paddle2ONNX with VisualDL service

VisualDL has deployed the model conversion tool on the website to provide services. You can click [Service Link] (https://www.paddlepaddle.org.cn/paddle/visualdl/modelconverter/) to perform online Paddle2ONNX model conversion.

![Paddle2ONNX](https://user-images.githubusercontent.com/22424850/226798785-33167569-4bd0-4b00-a5c0-5d6642cd6751.gif)

## Documents

- [model zoo](docs/en/model_zoo.md)
- [op list](docs/en/op_list.md)
- [update notes](docs/en/change_log.md)

## License
[Apache-2.0 license](https://github.com/PaddlePaddle/paddle-onnx/blob/develop/LICENSE).
