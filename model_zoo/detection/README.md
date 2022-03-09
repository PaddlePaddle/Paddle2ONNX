# 目标检测模型库

本文档中模型库均来源于PaddleDetection [release/2.3分支](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.3)，在下表中提供了部分已经转换好的模型，如有更多模型或自行模型训练导出需求，可参考 [PaddleDetection模型导出说明](https://github.com/PaddlePaddle/PaddleDetection/blob/develop/deploy/EXPORT_MODEL.md).
|模型名称|配置文件|模型大小|下载地址|说明|
| --- | --- | --- | --- | ---- |
|picodet|[picodet_l_640_coco.yml](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.3/configs/picodet/picodet_l_640_coco.yml)|618KB|[Paddle模型](https://bj.bcebos.com/paddle2onnx/model_zoo/picodet_l_640_coco.tar.gz) / [ONNX模型](https://bj.bcebos.com/paddle2onnx/model_zoo/picodet_l_640_coco.onnx)| 使用coco数据作为训练数据，80个分类，包括车、路、人等等 |


# ONNX模型推理示例

各模型的推理前后处理参考本目录下的infer.py，以PicoDet为例，如下命令即可得到推理结果

```bash
# 安装onnxruntime
pip3 install onnxruntime

# 下载PicoDet模型
wget https://bj.bcebos.com/paddle2onnx/model_zoo/picodet_l_640_coco.onnx

python3 infer.py \
    --model_path picodet_l_640_coco.onnx \
    --image_path ./images/hrnet_demo.jpg
```

你也可以使用Paddle框架进行推理验证

```bash
wget https://bj.bcebos.com/paddle2onnx/model_zoo/picodet_l_640_coco.tar.gz
tar xvf picodet_l_640_coco.tar.gz

python3 infer.py \
    --model_path ./picodet_l_640_coco/model \
    --image_path ./images/hrnet_demo.jpg \
    --use_paddle_predict True
```

执行命令后，在 ./outputs/ 下保存可视化结果。

ONNXRuntime 执行效果：

<div align="center">
    <img src="./images/onnx_hrnet_demo.jpg" width=500">
</div>

Paddle Inference 执行效果：

<div align="center">
    <img src="./images/onnx_hrnet_demo.jpg" width=500">
</div>