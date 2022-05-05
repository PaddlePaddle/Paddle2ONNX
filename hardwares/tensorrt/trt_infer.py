import onnx
import numpy as np
import trt_backend

model_file = "model.onnx"
model = onnx.load(model_file)
trt_engine = trt_backend.TrtEngine(model, max_batch_size=16)

data = np.random.rand(1, 80).astype("float32")

result = trt_engine.infer([data])
print(result[0])
