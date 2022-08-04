# RK模型分割模型部署指南
本文档介绍在RK系列芯片上部署PP_HumanSeg模型的步骤，具体包括：
1. 使用Paddle2ONNX将PaddleInference model转换为ONNX模型格式。 
2. 运行推理脚本获得推理结果。

## 模型转换
```text
# 进入目录
cd ./weights/onnx

# 下载ppseg模型
!wget https://paddleseg.bj.bcebos.com/dygraph/pp_humanseg_v2/portrait_pp_humansegv2_lite_256x144_inference_model_with_softmax.zip
!unzip portrait_pp_humansegv2_lite_256x144_inference_model_with_softmax.zip
# 转换模型
paddle2onnx --model_dir ./ppseg_lite_portrait_398x224_with_softmax \
            --model_filename model.pdmodel \
            --params_filename model.pdiparams \
            --save_file ppseg_lite_portrait_192x192_with_softmax.onnx \
            --opset_version 11 \
            --enable_onnx_checker True \
            --input_shape_dict "{'x': [1, 3, 192, 192]}"
            
# 删除无用文件
rm -rf ppseg_lite_portrait_398x224_with_softmax
rm -rf ppseg_lite_portrait_398x224_with_softmax.tar.gz
```

## 运行
### 更多参数
执行以下命令
```text
python picodet_infer.py -h
```
### ONNX
推理执行以下命令
```text
python humanseg_infer.py
```

### RKNN for PC
推理执行以下命令
```text
 python humanseg_infer.py --backend_type rk_pc
```

### RKNN for Board
推理执行以下命令
```text
sudo -E python3 humanseg_infer.py --backend_type rk_board \
                                  --model_path ./weights/rknn/ppseg_lite_portrait_192x192_with_softmax.rknn 
```

## 踩坑
以下为坑的主要原因:
* 无论怎么设置，PP_HumanSeg的算子导出后，算子版本永远都是12。
* 并且这个模型在转换为rknn的时候是有bug的，无法指定输出节点，因此我们是无法自行完成模型推中结尾部分的后处理的，只能一次在npu上推理完成.
* model_zoo下的模型推理时会有很大的误差

解决方案：不合适就换模型，换成最新的V2版本就可以了，但是上面的坑仍然存在