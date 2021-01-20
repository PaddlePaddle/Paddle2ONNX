# FAQ
Q1: What does the information "The parameter normalized of multiclass_nms OP of Paddle is False, which has diff with ONNX" mean in converting a model?

A: This is a warning and model conversion will not be influenced. The operator fluid.layers.multiclass_nms in PaddlePaddle has a normalized parameter, representing if the iput box has done normalization, and if its value is False in your model(mostly Yolov3), the inference result may have diff with orignal model.

Q2: What does the information "Converting this model to ONNX need with static input shape, please fix input shape of this model" mean in converting a model?

A: This implies an error, and the input shape of the model should be fixed:

- If the model is originated from PaddleX, you can designate it with --fixed_input_shape=[Height,Width]. Details please refer to [Exporting models in PaddleX](https://github.com/PaddlePaddle/PaddleX/blob/develop/docs/deploy/export_model.md).
- If the model is originated from PaddleDetection, you can designate it with TestReader.inputs_def.image_shape=[Channel,Height,Width]. Details please refer to [Exporting models in PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection/blob/master/docs/advanced_tutorials/deploy/EXPORT_MODEL.md#%E8%AE%BE%E7%BD%AE%E5%AF%BC%E5%87%BA%E6%A8%A1%E5%9E%8B%E7%9A%84%E8%BE%93%E5%85%A5%E5%A4%A7%E5%B0%8F).
- If the network of the model is built manually, you can designate it in fluid.data(shape=[]) by setting the shape parameter to fix the input size.
