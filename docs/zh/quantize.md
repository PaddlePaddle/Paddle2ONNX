# PaddleSlim量化模型转为ONNX格式
目前PaddleSlim有两种常用的量化方法，离线量化(PTQ)和量化训练(QAT)，Paddle2ONNX只支持离线量化模型导出为ONNX格式，并用ONNXRuntime在CPU上进行加速推理，量化训练模型的导出暂未支持。  
在PaddleSlim中进行离线量化时请开启onnx_format和is_full_quantize开关，使用Paddle2ONNX转换时和非量化模型的导出方式一致，不用特殊配置。PaddleSlim量化demo和接口请查阅：[PaddleSlim离线量化demo](https://github.com/PaddlePaddle/PaddleSlim/tree/develop/demo/quant/quant_post)  
一个简单的量化配置说明如下：  
```
from paddle.fluid.contrib.slim.quantization import PostTrainingQuantization
from paddle import fluid
place = fluid.CPUPlace()
exe = fluid.Executor(place)
ptq = PostTrainingQuantization(
    executor=exe,
    model_dir=model_path, # 待量化模型的存储路径
    sample_generator=val_reader, # 输入数据reader
    batch_size=batch_size,
    batch_nums=batch_nums,
    algo=algo, # 量化算法支持hist，KL，mse等多种算法
    quantizable_op_type=quantizable_op_type,
    is_full_quantize=True, # 是否开启全量化，如需导出为ONNX格式，请将此配置打开
    optimize_model=False, # 量化前是否先对模型进行优化，如需导出为ONNX格式，请关闭此配置
    onnx_format=True, # 量化OP是否为ONNX格式，如需导出为ONNX格式，请将此配置打开
    skip_tensor_list=skip_tensor_list,
    is_use_cache_file=is_use_cache_file)
ptq.quantize() # 对模型进行量化
ptq.save_quantized_model(int8_model_path) # 保存量化后的模型，int8_model_path为量化模型的保存路径
```
## FAQ
1. 模型导出时提示fake_quantize_dequantize_* 或fake_quantize_* 等OP不支持  
答：使用PaddleSlim离线量化时没有开启onnx_format开关，请开启onnx_format和is_full_quantize开关之后重新导出量化模型。  
2. 量化模型使用ONNXRuntime推理时精度相比Paddle-TRT、MKLDNN或者Paddle原生有明显下降  
答：如遇到使用ONNXRuntime推理时精度下降较多，可先用PaddleInference原生推理(不开启任何优化)验证是否量化模型精度本身就较低，如只有ONNXRuntime精度下降，请在终端执行：lscpu 命令，在Flags处查看机器是否支持avx512-vnni指令集，因为ONNXRuntime对量化模型推理还不是特别完备的原因，在不支持avx512-vnni的机器上可能会存在数据溢出的问题导致精度下降。  
可进一步使用如下脚本确认是否为该原因导致的精度下降，在使用ONNXRuntime推理时将优化全都关闭，然后再测试精度是否不再下降，如果还是存在精度下降问题，请提ISSUE给我们。
```
import onnxruntime as ort
providers = ['CPUExecutionProvider'] # 指定用CPU进行推理
sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL # 关闭所有的优化
sess = ort.InferenceSession(model_path, providers=providers, sess_options=sess_options) # model_path为ONNX模型
pred_onnx = sess.run(None, input_dict) # 进行推理
```
3. 量化模型相比于非量化模型没有加速，反而变慢了  
答：模型量化相比非量化模型变慢了可以从以下几个原因分析：  
(1) 检查机器是否支持avx512_vnni指令，在支持avx512_vnni的CPU上精度和加速效果最好  
(2) 检查是否在CPU推理，当前导出的ONNX模型仅支持使用ONNXRuntime在CPU上进行推理加速  
(3) 量化模型对计算量大的Conv或MatMul等OP加速明显，如果模型中Conv或MatMul的计算量本身很小，那么量化可能并不会带来推理加速  
(4) 使用如下命令获得ONNXRuntime优化后的模型optimize_model.onnx，然后使用VisualDl或netron等可视化工具可视化模型，检查以下两项：  
1). 检查原模型中的Conv、MatMul和Mul等OP是否已经优化为QLinearConv、QLinearMatMul和QLinearMul等量化相关OP  
2). 观察优化后的模型中量化OP是否被非量化OP分开得很散，多个量化OP链接在一起，不需量化和反量化获得的加速效果最明显，如果是激活函数导致的QLinearConv等量化OP被分开，推荐将激活函数替换为Relu或LeakyRelu再进行测试  
```
import onnxruntime as ort
providers = ['CPUExecutionProvider'] # 指定用CPU进行推理
sess_options = ort.SessionOptions()
sess_options.optimized_model_filepath = "./optimize_model.onnx" # 生成ONNXRuntime优化后的图，保存为optimize_model.onnx
sess = ort.InferenceSession(model_path, providers=providers, sess_options=sess_options) # model_path为ONNX模型
```
