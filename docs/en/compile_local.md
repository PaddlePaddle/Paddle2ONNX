# How to Compile and Install Paddle2ONNX Locally

The compilation and installation of Paddle2ONNX require ensuring that the environment meets the following requirements:

- cmake >= 3.16.0
- protobuf >= 4.22.0

## 1 Install on Linux/Mac

### 1.1 Install Protobuf

```bash
git clone https://github.com/protocolbuffers/protobuf.git
cd protobuf
git checkout v4.22.0
git submodule update --init
mkdir build_source && cd build_source
cmake ../cmake -DCMAKE_INSTALL_PREFIX=`pwd`/installed_protobuf_lib -Dprotobuf_BUILD_SHARED_LIBS=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON -Dprotobuf_BUILD_TESTS=OFF -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_STANDARD=14
make -j
make install

# set the library to environment
export PATH=${PWD}/installed_protobuf_lib/bin:${PATH}
```

### 1.2 Install Protobuf

```bash
python -m pip install --pre paddlepaddle -i https://www.paddlepaddle.org.cn/packages/nightly/cpu/
```

### 1.3 Install Paddle2ONNX

```bash
git clone https://github.com/PaddlePaddle/Paddle2ONNX.git
cd Paddle2ONNX
git submodule update --init
export PIP_EXTRA_INDEX_URL="https://www.paddlepaddle.org.cn/packages/nightly/cpu/"
python -m build
pip install dist/*.whl
```

If you are developing the Paddle2ONNX project locally, you can use `pip install -e .` to install it in editable mode.

## 2 Install on Windows

**Note that the prerequisite for compiling and installing Windows is that Visual Studio 2019 is already installed in the system**

### 2.1 Open Visual Studio Command Prompt

In the system menu, find **x64 Native Tools Command Prompt for VS 2019** and open it.

### 2.2 Install Protobuf

Note that the `-DCMAKE_INSTALL_PREFIX` in the following cmake command specifies your actual set path.

```bash
git clone https://github.com/protocolbuffers/protobuf.git
cd protobuf
git checkout v4.22.0
git submodule update --init --recursive
mkdir build
cd build
cmake -G "Visual Studio 16 2019"  -DCMAKE_INSTALL_PREFIX=%CD%\protobuf_install -Dprotobuf_MSVC_STATIC_RUNTIME=OFF -Dprotobuf_BUILD_SHARED_LIBS=OFF -Dprotobuf_BUILD_TESTS=OFF -Dprotobuf_BUILD_EXAMPLES=OFF ..
cmake --build . --config Release --target install
# set the library to environment
set PATH=%CD%\protobuf_install\bin;%PATH%
```

### 2.3 Install Paddle2ONNX

```bash
git clone https://github.com/PaddlePaddle/Paddle2ONNX.git
cd Paddle2ONNX
git submodule update --init
set PIP_EXTRA_INDEX_URL=https://www.paddlepaddle.org.cn/packages/nightly/cpu/
pip install setuptools wheel auditwheel auditwheel-symbols build
python -m build
pip install dist/*.whl
```
