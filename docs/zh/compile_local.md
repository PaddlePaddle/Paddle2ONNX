# 如何在本地编译并安装 Paddle2ONNX

Paddle2ONNX 的编译安装需要确保环境满足以下需求：

- cmake >= 3.16.0
- protobuf == 21.12

## 1 在 Linux/Mac 下安装

### 1.1 安装 Protobuf

```bash
git clone https://github.com/protocolbuffers/protobuf.git
cd protobuf
git checkout v21.12
git submodule update --init
mkdir build_source && cd build_source
cmake ../cmake -DCMAKE_INSTALL_PREFIX=`pwd`/installed_protobuf_lib -Dprotobuf_BUILD_SHARED_LIBS=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON -Dprotobuf_BUILD_TESTS=OFF -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_STANDARD=14
make -j
make install

# 将库路径添加到环境变量
export PATH=${PWD}/installed_protobuf_lib/bin:${PATH}
```

### 或者：

通过 apt 安装 Protobuf (Linux):

```bash
sudo apt install protobuf-compiler
```

通过 homebrew 安装 Protobuf (Mac):

```bash
brew install protobuf
```

### 1.2 安装 PaddlePaddle

```bash
python -m pip install --pre paddlepaddle -i https://www.paddlepaddle.org.cn/packages/nightly/cpu/
```

### 1.3 安装 Paddle2ONNX

```bash
git clone https://github.com/PaddlePaddle/Paddle2ONNX.git
cd Paddle2ONNX
git submodule update --init
export PIP_EXTRA_INDEX_URL="https://www.paddlepaddle.org.cn/packages/nightly/cpu/"
python -m build
pip install dist/*.whl
```

如果你是在本地开发 Paddle2ONNX 项目，可以使用 `pip install -e .` 命令以可编辑模式安装。

## 2 在 Windows 下安装

**注意，在 Windows 上编译安装的先决条件是系统已安装 Visual Studio 2019**

### 2.1 安装 Visual Studio 16 2019

1. 从[此链接](https://download.visualstudio.microsoft.com/download/pr/e7ffa30b-43a5-4afc-bf2a-2e3656a842e4/60b26131ac7b8c59f734a1e0c32cc9dc/vs_community.exe)下载 Visual Studio 16 2019 并运行 `vs_community.exe`。
2. 在 **工作负载** 选项卡下，勾选 **使用 C++ 的桌面开发**。
   - 注意：在安装详细信息 > 使用 C++ 的桌面开发 > 可选 中：**Live Share** 和 **Intellicode** 不是必需的，可以取消勾选。
3. 点击 **安装**

### 2.2 打开 Visual Studio 命令提示符

在系统菜单中，找到 **x64 Native Tools Command Prompt for VS 2019** 并打开。

### 2.3 安装 Protobuf

注意下面cmake命令中`-DCMAKE_INSTALL_PREFIX`指定为你实际设定的路径。

```bash
git clone https://github.com/protocolbuffers/protobuf.git
cd protobuf
git checkout v21.12
git submodule update --init --recursive
mkdir build
cd build
cmake -G "Visual Studio 16 2019"  -DCMAKE_INSTALL_PREFIX=%CD%\protobuf_install -Dprotobuf_MSVC_STATIC_RUNTIME=OFF -Dprotobuf_BUILD_SHARED_LIBS=OFF -Dprotobuf_BUILD_TESTS=OFF -Dprotobuf_BUILD_EXAMPLES=OFF ..
cmake --build . --config Release --target install
# 将库设置到环境变量
set PATH=%CD%\protobuf_install\bin;%PATH%
```

### 2.4 安装 Paddle2ONNX

```bash
git clone https://github.com/PaddlePaddle/Paddle2ONNX.git
cd Paddle2ONNX
git submodule update --init
set PIP_EXTRA_INDEX_URL=https://www.paddlepaddle.org.cn/packages/nightly/cpu/
pip install setuptools wheel auditwheel auditwheel-symbols build
python -m build
pip install dist/*.whl
```
