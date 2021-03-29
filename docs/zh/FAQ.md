# FAQ

Q1: 转换出错，提示 "Converting this model to ONNX need with static input shape, please fix input shape of this model"
- 在某些场景下，模型的输入大小需要固定才能使用Paddle2ONNX成功转换，原因在于PaddlePaddle与ONNX算子上的差异。
- 例如对于图像分类或目标检测模型而言，[-1, 3, -1, -1]被认为是动态的输入大小，而[1, 3, 224, 224]则是固定的输入大小（不过大多数时候batch维并不一定需要固定， [-1, 3, 224, 224]很可能就可以支持转换了)

- 如果模型是来自于PaddleX，参考此[文档导出](https://github.com/PaddlePaddle/PaddleX/blob/develop/docs/deploy/export_model.md)通过指定`--fixed_input_shape`固定大小
- 如果模型来自于PaddleDetection，参考此[文档导出](https://github.com/PaddlePaddle/PaddleDetection/blob/master/docs/advanced_tutorials/deploy/EXPORT_MODEL.md#%E8%AE%BE%E7%BD%AE%E5%AF%BC%E5%87%BA%E6%A8%A1%E5%9E%8B%E7%9A%84%E8%BE%93%E5%85%A5%E5%A4%A7%E5%B0%8F)
