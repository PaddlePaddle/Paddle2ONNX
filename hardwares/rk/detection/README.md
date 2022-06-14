# RK模型检测模型部署指南
本文档介绍在RK系列芯片上部署Paddle检测模型的步骤，具体包括：  
1. 使用Paddle2ONNX将PaddleInference model转换为ONNX模型格式。
2. 运行推理脚本获得推理结果。  

**RK PC模拟环境快速部署脚本**：bash quick_deploy.sh
## 模型转换
以picodet作为转换示例
```
# 下载picodet模型
wget https://bj.bcebos.com/paddle2onnx/model_zoo/picodet_s_320_coco.tar.gz
tar xvf picodet_s_320_coco.tar.gz
# 将Paddle模型导出为ONNX模型
paddle2onnx --model_dir ./picodet_s_320_coco --model_filename model.pdmodel --params_filename model.pdiparams --save_file picodet_s_320_coco.onnx --opset_version 12 --enable_onnx_checker True --input_shape_dict "{'image': [1, 3, 320, 320]}"
```
## 模型推理示例
以picodet检测模型为例
### 在RK PC模拟环境中进行推理
```
python deploy.py --model_file picodet_s_320_coco.onnx --image_path images/demo.jpg --backend_type rk_pc

# 运行结果
class_id:0, confidence:0.9487, left_top:[104.57,22.85],right_bottom:[256.42,388.99]
class_id:32, confidence:0.9492, left_top:[43.43,356.76],right_bottom:[86.72,401.42]
save result to: ./output_dir/rk_pc_demo.jpg
```
### 在RK部署平台上进行推理
在RK部署平台上部署推理需要进行三个简单步骤：
1. 在RK PC环境中运行推理，生成RK硬件上运行的rknn模型
2. 将模型的部署脚本拷贝到RK部署平台上
3. 运行部署程序
```
# step 1: 生成部署模型，模型后缀rknn
python deploy.py --model_file picodet_s_320_coco.onnx --image_path images/demo.jpg --backend_type rk_pc

# step 2：拷贝模型和脚本到部署平台
scp -r * xxx@ip:folder

# step 3：在部署平台上运行部署脚本
python3 deploy.py --model_file model.rknn --image_path images/demo.jpg --backend_type rk_hardware

# 运行结果
class_id:0, confidence:0.9473, left_top:[102.57,23.33],right_bottom:[257.92,387.79]
class_id:32, confidence:0.9482, left_top:[43.43,356.78],right_bottom:[86.72,401.44]
save result to: ./output_dir/rk_hardware_demo.jpg
```
### 使用ONNXRuntime进行推理
```
# 使用ONNXRuntime进行推理
python deploy.py --model_file picodet_s_320_coco.onnx --image_path images/demo.jpg --backend_type onnxruntime

# 运行结果
class_id:0, confidence:0.9487, left_top:[104.58,22.86],right_bottom:[256.41,389.04]
class_id:32, confidence:0.9492, left_top:[43.42,356.76],right_bottom:[86.73,401.43]
save result to: ./output_dir/onnxruntime_demo.jpg
```
### 使用Paddle进行推理
```
# 使用Paddle进行推理
python deploy.py --image_path images/demo.jpg --backend_type paddle --model_dir picodet_s_320_coco

# 运行结果
class_id:0, confidence:0.9487, left_top:[104.58,22.86],right_bottom:[256.41,389.04]
class_id:32, confidence:0.9492, left_top:[43.42,356.76],right_bottom:[86.73,401.43]
save result to: ./output_dir/paddle_demo.jpg
```
## 注意事项
1. RK尚不支持动态shape的输入，因此在使用Paddle2ONNX将Paddle模型转换为ONNX模型格式时需要输入input_shape
2. RK支持Opset version <= 12
3. RK相关API可参考文档： [RK API文档](https://github.com/rockchip-linux/rknn-toolkit2/blob/master/doc/Rockchip_User_Guide_RKNN_Toolkit2_CN-1.3.0.pdf)  
4. ONNXRuntime要求输入为NCHW，RK要求输入为NHWC
5. RK的PC模拟环境中，backend_type设置为rk_pc
6. 在RK部署芯片上，backend_type设置为rk_hardware

