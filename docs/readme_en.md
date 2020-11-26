# Paddle2ONNX

## Introduction

Paddle2ONNX enables users to convert models from PaddlePaddle to ONNX.

- Supported model format. Paddle2ONNX supports both dynamic and static computational graph of PaddlePaddle. For static computational graph, Paddle2ONNX converts PaddlePaddle models saved by API [save_inference_model](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/static/save_inference_model_cn.html#save-inference-model), for example [IPthon example](examples/tutorial.ipynb).For dynamic computational graph, it is now under experiment and more details will be released after the release of PaddlePaddle 2.0.
- Supported operaters. Paddle2ONNX can stably export models to ONNX Opset 9~11, and partialy support lower version Opset. More details please refer to [Operator list](docs/op_list.md).
- Supported models. You can find officially verified models by Paddle2ONNX in [model zoo](docs/model_zoo.md).

## Environment dependencies

### Configuration
     python >= 2.7  
     static computational graph: paddlepaddle >= 1.8.0
     dynamic computational graph: paddlepaddle >= 2.0.0
     onnx == 1.7.0 | Optional
## Installation

### Pip
    pip install paddle2onnx

### From source

     git clone https://github.com/PaddlePaddle/paddle2onnx.git
     python setup.py install

##  Usage
### Static computational graph
#### Using with command line
Uncombibined PaddlePaddle model(parameters saved in different files)

    paddle2onnx --model_dir paddle_model  --save_file onnx_file --opset_version 10 --enable_onnx_checker True

Combined PaddlePaddle model(parameters saved in one binary file)

    paddle2onnx --model_dir paddle_model  --model_filename model_filename --params_filename params_filename --save_file onnx_file --opset_version 10 --enable_onnx_checker True

#### Parameters
| parameters |Description |
|----------|--------------|
|--model_dir | the directory path of the paddlepaddle model saved by `paddle.fluid.io.save_inference_model`|
|--model_filename |**[Optional]** the model file name under the directory designated by`--model_dir`. Only needed when all the model parameters saved in one binary file. Default value None|
|--params_filename |**[Optonal]** the parameter file name under the directory designated by`--model_dir`. Only needed when all the model parameters saved in one binary file. Default value None|
|--save_file | the directory path for the exported ONNX model|
|--opset_version | **[Optional]** To configure the ONNX Opset version. Opset 9-11 are stably supported. Default value is 9.|
|--enable_onnx_checker| **[Optional]**  To check the validity of the exported ONNX model. It is suggested to turn on the switch. If set to True, onnx>=1.7.0 is required. Default value is False|
|--version |**[Optional]** check the version of paddle2onnx |

- Two types of PaddlePaddle models 
   - Combined model, parameters saved in one binary file. --model_filename and --params_filename represents the file name and parameter name under the directory designated by --model_dir. --model_filename and --params_filename are valid only with parameter --model_dir.
   - Uncombined model, parameters saved in different files. Only --model_dir is needed，which contains '\_\_model\_\_' file and the seperated parameter files.


#### IPython tutorials

- [convert to ONNX from static computational graph](examples/tutorial.ipynb)

### Dynamic computational graph

Under experiment now, tutorials will be provided after the release of PaddlePaddle 2.0.

## Relative documents

- [model zoo](docs/model_zoo.md)
- [op list](docs/op_list.md)
- [update notes](docs/change_log.md)

## License
[Apache-2.0 license](https://github.com/PaddlePaddle/paddle-onnx/blob/develop/LICENSE).