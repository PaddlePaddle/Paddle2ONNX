# 图像分类模型库

本文档中模型库均来源于PaddleCls [release/2.3分支](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.3/)，在下表中提供了部分已经转换好的模型，如有更多模型或自行模型训练导出需求，可参见[ImageNet 预训练模型库
](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.3/docs/zh_CN/algorithm_introduction/ImageNet_models.md).

|模型名称|模型大小|下载地址|说明|
| --- | --- | --- | ---- |
|ResNet50|102.5M|[推理模型](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/inference/ResNet50_infer.tar) / [ONNX模型](model.onnx)| 使用ImageNet数据作为训练数据，1000个分类，包括车、路、人等等 |
|PPLCNet|11.9M|[推理模型](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/inference/PPLCNet_x1_0_infer.tar) / [ONNX模型](model.onnx)| 使用ImageNet数据作为训练数据，1000个分类，包括车、路、人等等 |
|MobileNetV2|14.2M|[推理模型](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/inference/MobileNetV2_infer.tar) / [ONNX模型](model.onnx)| 使用ImageNet数据作为训练数据，1000个分类，包括车、路、人等等 |
|MobileNetV3_small|11.9M|[推理模型](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/inference/MobileNetV3_small_x1_0_infer.tar) / [ONNX模型](model.onnx)| 使用ImageNet数据作为训练数据，1000个分类，包括车、路、人等等 |


# 模型推理预测

- 环境依赖
    - paddlepaddle >= 2.0.2
    - paddle2onnx >= 0.9
    - onnxruntime >= 1.9.0

- 下载模型

以ResNet50为例：

在[图像分类模型库](#图像分类模型库)，下载ResNet50的推理模型和ONNX模型

```bash
wget -nc  -P ./inference https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/inference/ResNet50_infer.tar
cd ./inference && tar xf ResNet50_infer.tar && cd ..

wget -nc  -P ./inference https://paddleocr.bj.bcebos.com/paddle2onnx/class_models/onnx/ResNet50_infer.tar
cd ./inference && tar xf ResNet50_infer.tar && cd ..
```

其中ONNX模型，也可以通过使用 Paddle2ONNX 将BiseNet的推理模型转换为ONNX格式，执行如下命令即可：

```bash
paddle2onnx --model_dir=./inference/ResNet50_infer \
--model_filename=inference.pdmodel \
--params_filename=inference.pdiparams \
--save_file=./inference/ResNet50_infer/model.onnx \
--opset_version=11 \
--enable_onnx_checker=True
```

执行完毕后，ONNX 模型会被分别保存在 `./inference/ResNet50_infer/`路径下

- 推理预测

ONNX模型测试步骤如下：

- Step1：初始化`ONNXRuntime`库并配置相应参数, 并进行预测
- Step2：`ONNXRuntime`预测结果和`Paddle Inference`预测结果对比

使用ImageNet验证集中的一张[图片](./images/ILSVRC2012_val_00000010.jpeg)


在本目录下，我们提供了`infer.py`脚本进行预测，执行如下命令即可：

```bash
python3.7 infer.py \
    --model_path ./inference/ResNet50_infer/inference \
    --onnx_path ./inference/ResNet50_infer/model.onnx \
    --image_path ./images/ILSVRC2012_val_00000010.jpeg
```

执行命令后在终端会打印出预测的识别信息如下。

```
Paddle inference top-5 results:
class id(s): [153, 332, 229, 265, 196], score(s): [0.43, 0.30, 0.08, 0.05, 0.05], label_name(s): ['Maltese dog, Maltese terrier, Maltese', 'Angora, Angora rabbit', 'Old English sheepdog, bobtail', 'toy poodle', 'miniature schnauzer']
ONNXRuntime top-5  results:
class id(s): [153, 332, 229, 265, 196], score(s): [0.43, 0.30, 0.08, 0.05, 0.05], label_name(s): ['Maltese dog, Maltese terrier, Maltese', 'Angora, Angora rabbit', 'Old English sheepdog, bobtail', 'toy poodle', 'miniature schnauzer']
```

由终端输出结果可见，`Paddle Inference`预测结果与`ONNXRuntime`引擎的结果完全一致。
