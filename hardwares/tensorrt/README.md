# TensorRT Python部署 Paddle 模型

## 一.环境搭建
###1.TensorRT 包下载安装
在[官网](https://developer.download.nvidia.cn/compute/redist/nvidia-tensorrt)下载符合自己机器的 TensorRT 包，并安装 whl 包
```
pip install TensorRT-xxx.whl
```

### 2.安装其他python依赖包
```
# 安装 pycude
pip install pycuda

#安装 onnx
pip install onnx

#安装 numpy
pip install numpy
```

## 二. FLOAT 模型推理部署
只需要加载目录下封装好的 trt_backend 代码，调用里边的推理接口即可。简单示例如下:
```python
import trt_backend
import numpy as np
import paddle2onnx
import paddle
paddle.set_device('cpu')
paddle_model = paddle.jit.load("model/model")
onnx_model = paddle2onnx.run_convert(paddle_model, opset_version=11)

# 优化 onnx 模型，可选
import onnxsim
onnx_model, optimized = onnxsim.simplify(onnx_model, input_shapes={"input": [12, 80]}, dynamic_input_shape=True, skipped_optimizers=["extract_constant_to_initializer"])↩
if not optimized:
    print("Optimize the onnx model failed")

# 使用 trt_backend 中接口，初始化 TensorRT 引擎
# 静态 shape
trt_engine = trt_backend.TrtEngine(onnx_model, max_batch_size=16)

# 动态 shape. 需要配置 shape_info (字典结构)，key 为输入名， value 为[该输入在推理中的最小形状(min_shape)、最常用的形状(opt_shape)、最大形状(max_shape)]
trt_engine = trt_backend.TrtEngine(onnx_model, shape_info={"input" :[[1, 80], [10, 80], [16, 80]]}, max_batch_size=16)

# FP16 加速
trt_engine = trt_backend.TrtEngine(onnx_model, shape_info={"input" :[[1, 80], [10, 80], [16, 80]]}, max_batch_size=16, precision_mode="fp16")

#准备数据
data = np.random.rand(8, 80).astype("float32")

# 进行推理.输入为 list，如有多输入需要都放在 list 中(如 trt_engine.infer([data1,data2]))
result = trt_engine.infer([data])

# 结果result为[output1, output2, ... outputn].
# 每一个输出节点对应list中一位numpy格式的结果
print(result[0])
```


## 三. 量化模型推理部署
PaddleSlim 的量化模型导出为 ONNX 并使用 TensorRT 进行部署有以下步骤：
1. PaddleSlim 量化模型，生成三个文件，分别是模型文件，如 model.pdmodel 或 __model__，权重文件，如 model.pdiparams 或__params__，和 scale 保存文件，如out_scale.txt

2. 使用 Paddle2ONNX 导出 TensorRT 部署的量化模型，导出时请设置 deploy_backend 为 tensorrt，导出成功后有两个文件，ONNX 模型和 calibration.cache 文件

3. 使用如下示例进行部署
```python
import trt_backend
import numpy as np
import paddle2onnx
import paddle

# 使用 trt_backend 中接口，初始化 TensorRT 引擎
# 静态 shape
trt_engine = trt_backend.TrtEngine(onnx_model, max_batch_size=16, precision_mode="int8")

# 动态 shape. 需要配置 shape_info(字典结构)，key为输入名， value为[该输入在推理中的最小形状(min_shape)、最常用的形状(opt_shape)、最大形状(max_shape)]，需要设置 precision_mode 为 "int8"
trt_engine = trt_backend.TrtEngine(onnx_model, shape_info={"input" :[[1, 80], [10, 80], [16, 80]]}, max_batch_size=16, precision_mode="int8", calibration_cache_file="calibration.cache")

#准备数据
data = np.random.rand(8, 80).astype("float32")

# 进行推理.输入为 list，如有多输入需要都放在 list 中(如 trt_engine.infer([data1,data2]))
result = trt_engine.infer([data])

# 结果 result 为 [output1, output2, ... outputn].
# 每一个输出节点对应list中一位numpy格式的结果
print(result[0])
```
