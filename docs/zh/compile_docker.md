# Docker编译安装Paddle2ONNX

Paddle2ONNX编译安装需要确保环境满足以下需求
- cmake >= 3.18.0
- protobuf == 3.16.0

注意：Paddle2ONNX产出的模型，在使用ONNX Runtime推理时，要求使用最新版本(1.10.0版本以及上），如若需要使用低版本(1.6~1.10之间），则需要将ONNX版本降至1.8.2，在执行完`git submodule update`后，执行如下命令，然后再进行编译
```
cd Paddle2ONNX/third/onnx
git checkout v1.8.1
```

拉取manylinux镜像并创建容器

```bash
docker pull quay.io/pypa/manylinux2014_x86_64
docker run --name p2o_build  -d quay.io/pypa/manylinux2014_x86_64 
```

创建容器并运行

```bash
docker start p2o_build
docker exec -it p2o_build bash
```

