# Paddle2ONNX CPP 示例
本示例展示如何将 Paddle2ONNX 加入到你的CPP工程当中。
在 Linux 下运行本示例请执行如下步骤：
1. 下载 Paddle2ONNX 库并解压，注意请根据你的平台下载对应的库，建议下载最新版本。[Paddle2ONNX库下载地址](https://github.com/PaddlePaddle/Paddle2ONNX/releases)  
2. 修改 CMakeLists.txt  中 paddle2onnx_install_path 为你实际的解压地址  
3. 执行命令：mkdir build && cd build && cmake .. && make -j12
4. 成功后可以看到生成可执行文件：p2o_exec
5. 可用如下脚本测试转换：
```
# 下载模型
wget https://bj.bcebos.com/paddle2onnx/model_zoo/mobilenetv3.tar.gz
# 解压模型
tar -xf mobilenetv3.tar.gz
# 使用编译的可执行文件进行模型转换，成功后可在当前目录下看到生成的 model.onnx 文件
./p2o_exec mobilenetv3/inference.pdmodel mobilenetv3/inference.pdiparams model.onnx
```
