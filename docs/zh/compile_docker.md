# Docker编译安装Paddle2ONNX

## 1 拉取manylinux镜像

根据系统架构拉取不同的manylinux镜像

```bash
# Pull manylinux2014_x86_64
docker pull quay.io/pypa/manylinux2014_x86_64
docker create --name p2o_build -it quay.io/pypa/manylinux2014_x86_64 /bin/bash
# Pull manylinux2014_x86_64
docker pull quay.io/pypa/manylinux2014_aarch64
docker create --name p2o_build -it quay.io/pypa/manylinux2014_aarch64 /bin/bash
```

## 2 创建并进入容器

创建并进入 Docker 容器

```bash
docker start p2o_build
docker exec -it p2o_build /bin/bash
```

## 3 拉取 Paddle2ONNX 仓库

执行以下命令来拉取并初始化 Paddle2ONNX 仓库

```bash
git clone https://github.com/PaddlePaddle/Paddle2ONNX.git
cd Paddle2ONNX
git submodule init
git submodule update
```

## 4 获取 protobuf 依赖库

### 4.1 使用 protobuf 预编译库

执行以下命令来下载 protobuf 依赖库

```bash
source .github/workflows/scripts/download_protobuf.sh
```

### 4.2 下载并编译 protobuf 预编译库

执行以下命令来下载并编译 protobuf 预编译库

```bash
git clone https://github.com/protocolbuffers/protobuf.git
cd protobuf
git checkout v3.16.0
mkdir build_source && cd build_source
cmake ../cmake -DCMAKE_INSTALL_PREFIX=${PWD}/installed_protobuf_lib -Dprotobuf_BUILD_SHARED_LIBS=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON -Dprotobuf_BUILD_TESTS=OFF -DCMAKE_BUILD_TYPE=Release
make -j8
make install

# 将编译目录加入环境变量
export PATH=${PWD}/installed_protobuf_lib/bin:${PATH}
```

## 5 执行编译和安装

```bash
/opt/python/cp38-cp38/bin/pip install setuptools wheel auditwheel auditwheel-symbols build
cd /path/to/Paddle2ONNX
/opt/python/cp38-cp38/bin/python -m build
/opt/python/cp38-cp38/bin/pip install dist/*.whl
```
