import numpy as np
import trt_backend

model = "hifigan_csmsc.onnx"
import onnxsim
ori_model = onnx.load(model)
print("Optimize the input onnx model....")
optimized_model, optimized = onnxsim.simplify(ori_model, input_shapes={"logmel": [12, 80]}, dynamic_input_shape=True)
if not optimized:
    print("Optimize the onnx model failed!.")
trt_engine = trt_backend.TrtEngine(optimized_model, shape_info={"logmel" :[[227, 80], [308, 80], [544, 80]]}, max_batch_size=544*300, static_shape=False)


import numpy as np
data = np.load('input.npy')
batch = data.shape[0]
result = trt_engine.infer([data])
result[0] = result[0][:batch*300].reshape(batch*300, 1)

