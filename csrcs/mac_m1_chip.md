# Mac M1芯片下的编译安装

当在使用Mac M1时，并且我们安装了Anaconda时，由于Anaconda安装时使用的是`x86_64`架构，在执行`python setup.py install`后，会导致在python里面`import paddle2onnx`出错。因此可以采用如下方式来解决。

1. 将terminal切换为`i386`模式
```
arch -x86_64 /bin/bash --login
```

2. 编译安装protobuf
```
wget https://github.com/protocolbuffers/protobuf/releases/download/v3.16.0/protobuf-cpp-3.16.0.tar.gz
tar -xvf protobuf-cpp-3.16.0.tar.gz
cd protobuf-3.16.0
mkdir build_source && cd build_source
cmake ../cmake -Dprotobuf_BUILD_SHARED_LIBS=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON -Dprotobuf_BUILD_TESTS=OFF -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=${PWD}/install_dir
make -j8
make install

# 将protobuf添加到环境变量
export PATH=${PWD}/install_dir/bin/:${PATH}
```

3. 编译安装Paddle2ONNX
```
git clone https://github.com/PaddlePaddle/Paddle2ONNX.git
cd Paddle2ONNX
git checkout cpp

# 使用Anaconda具体环境的python进行编译安装
/Users/paddle/opt/anaconda3/envs/py37/bin/python setup.py install
```

在上述步骤后，之后即可在Anaconda环境中使用Paddle2ONNX。
