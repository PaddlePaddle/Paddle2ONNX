# 如何在本地编译并安装 Paddle2ONNX

Paddle2ONNX 的编译安装需要确保环境满足以下需求：

- cmake >= 3.16.0
- Python >= 3.10
- protobuf >= 34.0

## 1 在 Linux/Mac 下安装

### 1.1 安装系统依赖

通过系统包管理器安装 protobuf、glog 和 pybind11。

通过 homebrew 安装 (Mac):

```bash
brew install protobuf glog pybind11
```

通过 apt 安装 (Linux):

```bash
sudo apt install protobuf-compiler libprotobuf-dev libgoogle-glog-dev pybind11-dev
```

### 1.2 安装 PaddlePaddle

```bash
python -m pip install --pre paddlepaddle -i https://www.paddlepaddle.org.cn/packages/nightly/cpu/
```

### 1.3 安装 Paddle2ONNX

```bash
git clone https://github.com/PaddlePaddle/Paddle2ONNX.git
cd Paddle2ONNX
uv sync
uv run python setup.py build
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
git checkout v34.0
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
uv sync
uv run python setup.py build
pip install dist/*.whl
```
