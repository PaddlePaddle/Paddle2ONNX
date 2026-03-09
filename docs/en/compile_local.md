# How to Compile and Install Paddle2ONNX Locally

The compilation and installation of Paddle2ONNX require ensuring that the environment meets the following requirements:

- cmake >= 3.16.0
- Python >= 3.10
- protobuf >= 34.0

## 1 Install on Linux/Mac

### 1.1 Install System Dependencies

Install protobuf, glog, and pybind11 via your system package manager.

On Mac (via Homebrew):

```bash
brew install protobuf glog pybind11
```

On Linux (via apt):

```bash
sudo apt install protobuf-compiler libprotobuf-dev libgoogle-glog-dev pybind11-dev
```

### 1.2 Install PaddlePaddle

```bash
python -m pip install --pre paddlepaddle -i https://www.paddlepaddle.org.cn/packages/nightly/cpu/
```

### 1.3 Install Paddle2ONNX

```bash
git clone https://github.com/PaddlePaddle/Paddle2ONNX.git
cd Paddle2ONNX
uv sync
uv run python setup.py build
pip install dist/*.whl
```

If you are developing the Paddle2ONNX project locally, you can use `pip install -e .` to install it in editable mode.

## 2 Install on Windows

**Note that the prerequisite for compiling and installing Windows is that Visual Studio 2019 is already installed in the system**
### 2.1 Install Visual Studio 16 2019
1. Download Visual Studio 16 2019 from [this link](https://download.visualstudio.microsoft.com/download/pr/e7ffa30b-43a5-4afc-bf2a-2e3656a842e4/60b26131ac7b8c59f734a1e0c32cc9dc/vs_community.exe) and run `vs_community.exe`.
2. Under the **Workloads** tab, select the checkbox for **Desktop development with C++**.
   - Note: Under Installation Details > Desktop Development with C++ > Optional: **Live Share** and **Intellicode** are unnecessary, feel free to uncheck these boxes.
3. Click **Install**

### 2.2 Open Visual Studio Command Prompt

In the system menu, find **x64 Native Tools Command Prompt for VS 2019** and open it.

### 2.3 Install Protobuf

Note that the `-DCMAKE_INSTALL_PREFIX` in the following cmake command specifies your actual set path.

```bash
git clone https://github.com/protocolbuffers/protobuf.git
cd protobuf
git checkout v34.0
mkdir build
cd build
cmake -G "Visual Studio 16 2019"  -DCMAKE_INSTALL_PREFIX=%CD%\protobuf_install -Dprotobuf_MSVC_STATIC_RUNTIME=OFF -Dprotobuf_BUILD_SHARED_LIBS=OFF -Dprotobuf_BUILD_TESTS=OFF -Dprotobuf_BUILD_EXAMPLES=OFF ..
cmake --build . --config Release --target install
# set the library to environment
set PATH=%CD%\protobuf_install\bin;%PATH%
```

### 2.4 Install Paddle2ONNX

```bash
git clone https://github.com/PaddlePaddle/Paddle2ONNX.git
cd Paddle2ONNX
uv sync
uv run python setup.py build
pip install dist/*.whl
```
