# 用python版TensorRT部署onnx模型

## 一.环境搭建
###1.TensorRT包下载安装
在[官网](https://developer.download.nvidia.cn/compute/redist/nvidia-tensorrt)下载符合自己机器的TensorRT包，并安装whl包
```
pip install TensorRT-xxx.whl
```

### 2.安装其他python依赖包
```
# 安装pycude
pip install pycuda

#安装onnx
pip install onnx

#安装numpy
pip install numpy
```

## 二. 推理部署
只需要加载目录下封装好的trt_backend代码，调用里边的推理接口即可。简单示例如下:
```python
import trt_backend
import numpy as np
import paddle2onnx
import paddle
paddle.set_device('cpu')
paddle_model = paddle.jit.load("model/model")
onnx_model = paddle2onnx.run_convert(paddle_model, opset_version=11)

# 优化onnx模型，可选
import onnxsim
onnx_model, optimized = onnxsim.simplify(onnx_model, input_shapes={"input": [12, 80]}, dynamic_input_shape=True, skipped_optimizers=["extract_constant_to_initializer"])↩
if not optimized:
    print("Optimize the onnx model failed")

# 使用trt_backend中接口，初始化TensorRT引擎
# 静态shape
trt_engine = trt_backend.TrtEngine(onnx_model, max_batch_size=16, static_shape=True)

# 动态shape. 需要配置shape_info(字典结构)，key为输入名， value为[该输入在推理中的最小形状(min_shape)、最常用的形状(opt_shape)、最大形状(max_shape)]
trt_engine = trt_backend.TrtEngine(onnx_model, shape_info={"input" :[[1, 80], [10, 80], [16, 80]]}, max_batch_size=16, static_shape=False)

#准备数据
data = np.random.rand(8, 80).astype("float32")

# 进行推理.输入为list，如有多输入需要都放在list中(如trt_engine.infer([data1,data2]))
result = trt_engine.infer([data])

# 结果result为[output1, output2, ... outputn]. 
# 每一个输出节点对应list中一位numpy格式的结果
print(result[0])
```
