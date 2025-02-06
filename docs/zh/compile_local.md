# 如何在本地编译并安装 Paddle2ONNX

Paddle2ONNX 的编译安装至少需要确保环境满足以下需求：

- cmake >= 3.16.0
- protobuf == 4.22.0

## 在 Linux/Mac 下编译并安装

### 安装 Protobuf

```bash
git clone https://github.com/protocolbuffers/protobuf.git
cd protobuf
git checkout v4.22.0
git submodule update --init
mkdir build_source && cd build_source
cmake ../cmake -DCMAKE_INSTALL_PREFIX=`pwd`/installed_protobuf_lib -Dprotobuf_BUILD_SHARED_LIBS=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON -Dprotobuf_BUILD_TESTS=OFF -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_STANDARD=14
make -j
make install

# 将编译目录加入环境变量
export PATH=${PWD}/installed_protobuf_lib/bin:${PATH}
```

### 安装PaddlePaddle
```bash
python -m pip install --pre paddlepaddle -i https://www.paddlepaddle.org.cn/packages/nightly/cpu/
```

### 安装Paddle2ONNX

```bash
git clone https://github.com/PaddlePaddle/Paddle2ONNX.git
cd Paddle2ONNX
git submodule update --init
export PIP_EXTRA_INDEX_URL="https://www.paddlepaddle.org.cn/packages/nightly/cpu/"
python -m build
pip install dist/*.whl
```

如果你是在本地开发　Paddle2ONNX 项目，推荐使用 `pip install -e .` 命令，以 editable mode 来安装。

## Windows编译安装

注意Windows编译安装先验条件是系统中已安装好Visual Studio 2019

### 打开VS命令行工具

系统菜单中，找到**x64 Native Tools Command Prompt for VS 2019**打开

### 安装Protobuf

注意下面cmake命令中`-DCMAKE_INSTALL_PREFIX`指定为你实际设定的路径

```bash
git clone https://github.com/protocolbuffers/protobuf.git
cd protobuf
git checkout v4.22.0
git submodule update --init --recursive
mkdir build
cd build
cmake -G "Visual Studio 16 2019"  -DCMAKE_INSTALL_PREFIX=%CD%\protobuf_install -Dprotobuf_MSVC_STATIC_RUNTIME=OFF -Dprotobuf_BUILD_SHARED_LIBS=OFF -Dprotobuf_BUILD_TESTS=OFF -Dprotobuf_BUILD_EXAMPLES=OFF ..
cmake --build . --config Release --target install
# 设置环境变量
set PATH=%CD%\protobuf_install\bin;%PATH%
```

### 安装Paddle2ONNX

```bash
git clone https://github.com/PaddlePaddle/Paddle2ONNX.git
cd Paddle2ONNX
git submodule update --init
set PIP_EXTRA_INDEX_URL=https://www.paddlepaddle.org.cn/packages/nightly/cpu/
pip install setuptools wheel auditwheel auditwheel-symbols build
python -m build
pip install dist/*.whl
```
