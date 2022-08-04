# RK模型OCR模型部署指南
本文档介绍在RK系列芯片上部署OCR模型的步骤，具体包括：
1. 使用Paddle2ONNX将PaddleInference model转换为ONNX模型格式。 
2. 运行推理脚本获得推理结果。

## 模型转换
```text
# 进入目录
cd ./weights/onnx

# 下载PP-OCRv2模型
wget https://bj.bcebos.com/paddle2onnx/model_zoo/ch_PP-OCRv2_det_infer.tar
tar xvf ch_PP-OCRv2_det_infer.tar

pip install onnx==1.7.0

# 转换模型
paddle2onnx --model_dir ./ch_PP-OCRv2_det_infer \
            --model_filename inference.pdmodel \
            --params_filename inference.pdiparams \
            --save_file PP_OCR_v2_det.onnx \
            --opset_version 11 \
            --enable_onnx_checker True \
            --input_shape_dict "{'x': [1, 3, 960, 960]}"
            
# 删除无用文件
rm -rf ch_PP-OCRv2_det_infer.tar
rm -rf ch_PP-OCRv2_det_infer
```

## 运行
### 更多参数
执行以下命令
```text
python PP_OCR_infer.py -h
```
### ONNX
推理执行以下命令
```text
python PP_OCR_infer.py
```

### RKNN for PC
推理执行以下命令
```text
python PP_OCR_infer.py --backend_type rk_pc
```

### RKNN for Board
推理执行以下命令
```text
sudo -E python3 PP_OCR_infer.py --backend_type rk_board \
                                  --det_model_dir ./weights/rknn/PP_OCR_v2_det.rknn 
```

## 踩坑

解决方案：不合适就换模型，换成最新的V2版本就可以了，但是上面的坑仍然存在