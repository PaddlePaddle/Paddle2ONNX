# RK模型分割模型部署指南

本文档介绍在RK系列芯片上部署PP_HumanSeg模型的步骤，具体包括：

1. 使用Paddle2ONNX将PaddleInference model转换为ONNX模型格式。
2. 运行推理脚本获得推理结果。

## 模型转换

```text
# 下载ONNX模型
# 进入目录
cd ./weights
mkdir onnx
cd onnx
wget https://paddlelite-demo.bj.bcebos.com/onnx%26rknn2_model/portrait_pp_humansegv2_lite_256x144_inference_model_with_softmax.onnx

# 下载RKNN模型
# 进入目录
cd ./weights
mkdir rknn
cd rknn
wget https://paddlelite-demo.bj.bcebos.com/onnx%26rknn2_model/portrait_pp_humansegv2_lite_256x144_inference_model_with_softmax.rknn
```

## 运行

### 更多参数

执行以下命令

```text
python pp_humanseg_infer.py -h
```

### ONNX

推理执行以下命令

```text
python pp_humanseg_infer.py
```

### RKNN for PC

推理执行以下命令

```text
 python pp_humanseg_infer.py --backend_type rk_pc
```

### RKNN for Board

推理执行以下命令

```text
sudo -E python3 pp_humanseg_infer.py --backend_type rk_board \
                                  --model_path ./weights/rknn/ppseg_lite_portrait_192x192_with_softmax.rknn 
```

## 结果展示

输入图片
![输入图片](./images/before/PP_HumanSeg_demo_input.jpg)

输出图片
![输出图片](./images/after/PP_HumanSeg_demo_output_rk_pc.png)

数据表:

| 模型            | speed(ms) |
|---------------|-----------|
| ONNX          | 45        |
| rk_board(量化前) | 25        |

## 踩坑

以下为坑的主要原因:

* 无论怎么设置，PP_HumanSeg的算子导出后，算子版本永远都是12。
* 并且这个模型在转换为rknn的时候是有bug的，无法指定输出节点，因此我们是无法自行完成模型推中结尾部分的后处理的，只能一次在npu上推理完成.
* model_zoo下的模型推理时会有很大的误差

解决方案：不合适就换模型，换成最新的V2版本就可以了，但是上面的坑仍然存在

