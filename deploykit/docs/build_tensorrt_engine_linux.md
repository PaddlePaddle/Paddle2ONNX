# 快速上手TensorRT部署

##  1 基于Docker

### 1.1  配置环境

安装 TensorRT Docker镜像之前，首先需要安装[Docker](https://docs.docker.com/engine/install/)，如果需要使用GPU预测请安装[NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker)。

拉取镜像的命令：

- `<xx.yy>`指的是你需要拉取的Tensorrt 镜像版本，以`20.11`为例，手动替换`<xx.yy>`为`20.11`：

```
$ docker pull  nvcr.io/nvidia/tensorrt:<xx.yy>-py3

```

创建一个名为 `tesnorrt-onnx` 的Docker容器：

```
$ docker run -it --gpus=1 --name tensorrt-onnx  -v ~/paddle2onnx/:/paddle2onnx/ --net=host nvcr.io/nvidia/tensorrt:20.11-py3 /bin/bash
```

## 1.2 项目编译

打开项目路径，运行编译脚本

```
$ cd /paddle2onnx/deploykit/cpp/
$ git clone https://github.com/NVIDIA/TensorRT.git
$ sh scripts/tensorrt_build.sh --tensorrt_dir=/usr/lib/x86_64-linux-gnu/ --cuda_dir=/usr/local/cuda-11.1/targets/x86_64-linux/ --tensorrt_header=./TensorRT/
```

## 1.3 准备模型

以[PPYOLO](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.0-rc/configs/ppyolo/README_cn.md)为例：


模型文件的获取首先参考[EXPORT_MODEL](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.0-rc/docs/advanced_tutorials/deploy/EXPORT_MODEL.md)，将训练后模型（或预训练模型）导出Paddle inference模型:

```
$ git clone https://github.com/PaddlePaddle/PaddleDetection.git
$ cd PaddleDetection
$ pip install -r requirements.txt
$ python tools/export_model.py -c configs/ppyolo/ppyolo_mobilenet_v3_small.yml \
        --output_dir=./inference_model --exclude_nms \
        -o weights=https://paddlemodels.bj.bcebos.com/object_detection/ppyolo_mobilenet_v3_small.tar \
           TestReader.inputs_def.image_shape=[3,320,320]
```

调用[Paddle2ONNX](https://github.com/PaddlePaddle/Paddle2ONNX.git)从Paddle转换为TensorRT目前支持的ONNX格式，安装方式参考[Paddle2ONNX安装](https://github.com/PaddlePaddle/Paddle2ONNX/blob/develop/README_zh.md#%E5%AE%89%E8%A3%85)。

```
$ git clone https://github.com/Channingss/paddle2onnx.git
$ cd paddle2onnx/
$ git checkout deform
$ python setup.py install
```

**注意** 由于PPYOLO使用Resize算子的参数，目前TensorRT还不支持，所以这里提供了兼容TensorRT的脚本来导出ONNX模型：

```
$ cd ../../script/tensorrt/
$ python export_onnx.py -m ../../cpp/PaddleDetection/inference_model/ppyolo_mobilenet_v3_small/ \
    --model_filename __model__ --params_filename __params__ --opset_version 11 \
    -s ../../cpp/PaddleDetection/inference_model/ppyolo_mobilenet_v3_small/model.onnx
```

## 1.4 推理

上述编译后会生成针对不同模型库套件实现的Demo，以`ppdet_triton_infer`为例，可请求`ppyolo_onnx`的推理。

```
$ cd ../../cpp/
$ ./build/demo/ppdet_tensorrt_infer  --model_dir PaddleDetection/inference_model/ppyolo_mobilenet_v3_small/model.onnx  \
    --image  demo/tensorrt_inference/imgs/ppyolo_test.jpg \
    --cfg_file  PaddleDetection/inference_model/ppyolo_mobilenet_v3_small/infer_cfg.yml

```

|参数名称 | 含义 |
|---|---|
| --model_dir | ONNX模型文件的路径 |
| --cfg_file | PaddleDetection导出模型时的配置文件，具体可以参考[EXPORT_MODEL](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.0-rc/docs/advanced_tutorials/deploy/EXPORT_MODEL.md) |
| --image | 需要预测的单张图片的文件路径 |
| --toolkit | Paddle模型库组件的别称，可选det、ocr、seg、cls等 |

## 2 基于Ubuntu

## 3 基于Windows
