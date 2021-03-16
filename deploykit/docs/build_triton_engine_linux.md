# 基于Docker快速上手Triton部署

Triton的全称为Triton Inference Server，它提供了针对CPU和GPU优化的云和边缘推理解决方案。 Triton支持HTTP / REST和GRPC协议，该协议允许远程客户端请求服务器管理的任何模型进行推理。本文将介绍如何基于Docker快速将Paddle的PPYOLO模型部署到Triton服务上。

## 1 安装Triton Docker镜像

安装 Triton Docker镜像之前，首先需要安装[Docker](https://docs.docker.com/engine/install/)，如果需要使用GPU预测请安装[NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker)。

拉取镜像的命令：

- `<xx.yy>`指的是你需要拉取的Triton docker版本，目前支持`20.11`，所以请手动替换`<xx.yy>`为`20.11`。
- 镜像后缀为`-py3`为Triton的服务端（server），`-py3-clientsdk`为Triton的客户端（client）。

```
docker pull nvcr.io/nvidia/tritonserver:<xx.yy>-py3
docker pull nvcr.io/nvidia/tritonserver:<xx.yy>-py3-clientsdk
```

## 2 部署Triton服务端

### 2.1 准备模型库（Model Repository）

在启动Triton服务之前，我们需要准备模型库，主要包括模型文件，模型配置文件和标签文件等，模型库的概念和使用方法请参考[model_repository](https://github.com/triton-inference-server/server/blob/master/docs/model_repository.md)，模型配置文件涉及到需要关键的配置，更详细的使用方法可参考[model_configuration](https://github.com/triton-inference-server/server/blob/master/docs/model_configuration.md)。

以[PPYOLO](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.0-rc/configs/ppyolo/README_cn.md)为例：模型文件的获取首先参考[EXPORT_MODEL](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.0-rc/docs/advanced_tutorials/deploy/EXPORT_MODEL.md)将训练后模型（或预训练模型）导出Paddle inference模型，再调用[Paddle2ONNX](https://github.com/PaddlePaddle/Paddle2ONNX.git)从Paddle转换为Triton目前支持的ONNX格式。

PPYOLO预训练ONNX模型下载脚本：

```
cd /path/to/paddle2onnx/deploykit/resource/triton/
sh fetch_models.sh
```


### 2.2 启动Triton server服务

经过Triton的优化可以使用GPU提供极佳的推理性能，且可以在仅支持CPU的系统上工作。以上两种情况下，我们都可以使用上述的Triton Docker镜像部署。

#### 2.2.1 启动基于GPU的服务

使用以下命令对刚刚创建的PPYOLO模型库运行Triton服务。需要注意的是，必须安装[NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker)，Docker才能识别GPU。 --gpus = 1参数表明Triton可以使用1块GPU进行推理。

```
docker run --gpus=1 --rm -p8000:8000 -p8001:8001 -p8002:8002 -v /path/to/paddle2onnx/deploykit/resource/model_repository/:/model_repository/
 nvcr.io/nvidia/tritonserver:<xx.yy>-py3 tritonserver --model-repository=/model_repository/
```
启动Triton之后，您将在控制台上看到如下输出，显示服务器正在启动并加载模型。当您看到如下输出时，Triton准备接受推理请求。
```
+----------------------+---------+--------+
| Model                | Version | Status |
+----------------------+---------+--------+
| <model_name>         | <v>     | READY  |
| ..                   | .       | ..     |
| ..                   | .       | ..     |
+----------------------+---------+--------+
...
...
...
I1002 21:58:57.891440 62 grpc_server.cc:3914] Started GRPCInferenceService at 0.0.0.0:8001
I1002 21:58:57.893177 62 http_server.cc:2717] Started HTTPService at 0.0.0.0:8000
I1002 21:58:57.935518 62 http_server.cc:2736] Started Metrics Service at 0.0.0.0:8002
```

所有模型均应显示“ READY”状态，以指示它们已正确加载。如果模型无法加载，则状态将报告失败以及失败的原因。如果您的模型未显示在表中，请检查模型库和CUDA驱动程序的路径。

#### 2.2.2 启动仅支持CPU的服务

在没有GPU的系统上，请在不使用--gpus参数的情况下运行，其他参数与上述启动GPU部署服务的命令一致。

```
docker run --rm -p8000:8000 -p8001:8001 -p8002:8002 -v /path/to/paddle2onnx/deploykit/resource/model_repository/:/model_repository/
 nvcr.io/nvidia/tritonserver:<xx.yy>-py3 tritonserver --model-repository=/model_repository/
```

### 2.3 验证远程服务

使用Triton的ready接口来验证服务器和模型是否已准备好进行推断，在主机系统重使用curl访问服务器的状态。

```
$ curl -v localhost:8000/v2/health/ready

打印：
...
< HTTP/1.1 200 OK
< Content-Length: 0
< Content-Type: text/plain
```

## 3 PPYOLO Demo

### 3.1 创建包含Triton客户端SDK的docker容器

通过指定--rm创建一个临时的Docker容器

```
docker run -it --rm --net=host nvcr.io/nvidia/tritonserver:<xx.yy>-py3-clientsdk -v /path/to /paddle2onnx/:/paddle2onnx/ /bin/bash/
```

### 3.2 项目编译

打开项目路径，运行编译脚本

```
$ cd /paddle2onnx/deploykit/cpp/
$ sh scripts/triton_build.sh --triton_client=/workspace/install/
```
### 3.3 请求服务端推理

上述编译后会生成针对不同模型库套件实现的Demo，以`ppdet_triton_infer`为例，可请求`ppyolo_onnx`的推理。

```
$ ./build/demo/ppdet_triton_infer --image /paddle2onnx/deploykit/resource/triton/imgs/ppyolo_test.jpg  --cfg_file /paddle2onnx/deploykit/resource/triton/client/ppyolo/infer_cfg.yml --url localhost:8000 --model_name ppyolo_onnx
```

|参数名称 | 含义 |
|---|---|
| --model_name | 模型名称|
| --url | 服务的远程IP地址+端口，如：localhost:8000|
| --model_version | 模型版本 |
| --cfg_file | PaddleDetection导出模型时的配置文件，具体可以参考[EXPORT_MODEL](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.0-rc/docs/advanced_tutorials/deploy/EXPORT_MODEL.md) |
| --image | 需要预测的单张图片的文件路径 |
| --image_list | .txt文件，每一行是一张图片的文件路径 |
| --toolkit | Paddle模型库组件的别称，可选det、ocr、seg、cls等 |
