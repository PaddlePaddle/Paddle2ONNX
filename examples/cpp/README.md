## Paddle2ONNX Linux x64 静态库使用步骤

### 安装Protobuf
```
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

### Paddle2ONNX 静态库库编译

- 编译paddle2onnx.a  
```bash
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=${PWD}/paddle2onnx-linux-x64 -DWITH_STATIC=ON
make -j8
make install
```  
- 打包所有依赖库
```bash
cp ./paddle2onnx/proto/libp2o_paddle_proto.a paddle2onnx-linux-x64/lib
cp {PATH-TO-YOUR}/installed_protobuf_lib/lib/libprotobuf.a paddle2onnx-linux-x64/lib
cp {PATH-TO-YOUR}/installed_protobuf_lib/lib/libprotoc.a paddle2onnx-linux-x64/lib
```
- ar合并依赖库   

首先将库拷贝到，exmaples/cpp目录下： 
```bash
cp -r paddle2onnx-linux-x64 ../examples/cpp
```
然后执行合并静态库脚本，得到libpaddle2onnx_bundled.a。在编译可执行文件需要用的是libpaddle2onnx.a和libpaddle2onnx_bundled.a  
```bash
cd ../examples/cpp
sh ./paddle2onnx_bundled-linux-x64.sh
```
```bash
+ ar -M
+ ls -lh paddle2onnx-linux-x64/lib/libpaddle2onnx_bundled.a
-rw-r--r-- 1 root root 22M Dec 27 03:50 paddle2onnx-linux-x64/lib/libpaddle2onnx_bundled.a
```

### 编译示例
在examples/cpp目录下  
```bash  
mkdir build && cd build
cmake ..
make -j
```
成功后可以看到生成可执行文件：p2o_exec，可用如下脚本测试转换：
```
# 下载模型
wget https://bj.bcebos.com/paddle2onnx/model_zoo/mobilenetv3.tar.gz
# 解压模型
tar -xf mobilenetv3.tar.gz
# 使用编译的可执行文件进行模型转换，成功后可在当前目录下看到生成的 model.onnx 文件
./p2o_exec mobilenetv3/inference.pdmodel mobilenetv3/inference.pdiparams model.onnx
```
通过ldd查看p2o_exec，可以发现，它已经不依赖paddle2onnx，只依赖系统库。因此可以放在其他linux x64系统上运行。
```bash
 ldd p2o_exec
	linux-vdso.so.1 (0x00007ffe0690f000)
	libstdc++.so.6 => /usr/lib/x86_64-linux-gnu/libstdc++.so.6 (0x00007ff95958e000)
	libm.so.6 => /lib/x86_64-linux-gnu/libm.so.6 (0x00007ff9591f0000)
	libgcc_s.so.1 => /lib/x86_64-linux-gnu/libgcc_s.so.1 (0x00007ff958fd8000)
	libc.so.6 => /lib/x86_64-linux-gnu/libc.so.6 (0x00007ff958be7000)
	/lib64/ld-linux-x86-64.so.2 (0x00007ff959916000)
``` 
### CMakeLists需要注意的问题  
编译示例完整的cmake配置，请参考[CMakeLists.txt](./CMakeLists.txt)。在编写CMakeLists时需要注意，由于paddle2onnx当前的op注册逻辑，会导致编译得到的libpaddle2onnx.a静态库在被链接进可执行文件时，无法初始化op注册相关的全局变量，从而无法正常使用。在linux x64下，可以通过'-Wl,--whole-archive'参数来强制链接器加载完整的libpaddle2onnx.a来解决这个问题。具体操作如下。
```cmake
TARGET_LINK_LIBRARIES(${PROJECT_NAME} ${paddle2onnx_lib})
TARGET_LINK_LIBRARIES(${PROJECT_NAME} ${paddle2onnx_bundled_lib})
set_target_properties(${PROJECT_NAME} PROPERTIES LINK_FLAGS 
                      "-Wl,--whole-archive ${paddle2onnx_lib} -Wl,-no-whole-archive") 
```
