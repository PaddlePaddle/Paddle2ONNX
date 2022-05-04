# DeployKit

DeployKit通过不同IR集成多种硬件推理后端的支持，目前包括
- ONNXRuntime
- TensorRT

编译
```
git clone https://github.com/PaddlePaddle/Paddle2ONNX.git
cd Paddle2ONNX
git checkout deploytkit
git submodule init
git submodule update
mkdir build
cd build
# 获取TensorRT/ONNXRuntime依赖库于本地并解压

cmake .. -DBUILD_DEPLOYKIT=ON \
         -DENABLE_ORT_BACKEND=ON \
         -DORT_DIRECTORY=${PWD}/onnxruntime-linux-x64-1.11.0 \
         -DENABLE_TRT_BACKEND=ON \
         -DTRT_DIRECTORY=${PWD}/TensorRT-8.4.0.6 \
         -DCUDA_DIRECTORY=/usr/local/cuda-10.2 \
         -DBUILD_DEMO=ON \
         -DENABLE_PADDLE_FRONTEND=ON
```
如若编译中提示cudnn的关联错误，请先将cudnn对应版本的lib目录路径导入到LD_LIBRARY_PATH
