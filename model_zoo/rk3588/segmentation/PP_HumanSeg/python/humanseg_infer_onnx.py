import cv2
import onnxruntime as rt

from utils.humanseg_tool import *

onnx_model_path = '../weights/onnx/model.onnx'
sess = rt.InferenceSession(onnx_model_path)
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name

target_size = (192, 192)

if True:
    raw_frame = cv2.imread("../images/before/human_image.jpg")
    pre_shape = raw_frame.shape[0:2][::-1]
    frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGRA2RGB)
    frame = preprocess(frame, target_size)
    pred = sess.run(
        [label_name],
        {input_name: frame.astype(np.float32)}
    )[0]
    pred = pred[0]
    print(np.array(pred).shape)
    raw_frame = resize(raw_frame, target_size)
    image = display_masked_image(pred, raw_frame)
    image = resize(image, target_size=pre_shape)
    cv2.imwrite('../images/after/ONNXResult.jpg', image)
