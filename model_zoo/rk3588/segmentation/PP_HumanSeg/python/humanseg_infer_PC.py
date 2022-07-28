from utils.rknn_config import RKNNConfigPC
import argparse
import os
import cv2
import numpy as np
from utils.humanseg_tool import *

np.set_printoptions(suppress=True)


def parse_args():
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument(
        '--model_path',
        dest='model_path',
        type=str,
        default='../weights/onnx/pp_humansegv2_lite_without_argmax.onnx',
        help="while use_paddle_predict, this means directory path of paddle model. Other wise, this means path of "
             "onnx model file.")
    parser.add_argument(
        '--image_path',
        dest='image_path',
        type=str,
        default="../images/before/human_image.jpg",
        help='The directory or path or file list of the images to be predicted.')
    parser.add_argument(
        '--use_paddle_predict',
        type=bool,
        default=False,
        help="If use paddlepaddle to predict, otherwise use onnxruntime to predict."
    )
    parser.add_argument(
        '--bg_image_path',
        dest='bg_image_path',
        help='Background image path for replacing. If not specified, a white background is used',
        type=str)
    parser.add_argument(
        '--save_dir',
        dest='save_dir',
        help='The directory for saving the inference results',
        type=str,
        default='../images/after/')
    return parser.parse_args()


if __name__ == "__main__":
    FLAGS = parse_args()
    onnx_model_path = FLAGS.model_path
    rknn_config = RKNNConfigPC(onnx_path=onnx_model_path, export_path="../weights/rknn/export.rknn")
    rknn = rknn_config.create_rknn()
    # sess = rt.InferenceSession(onnx_model_path)
    # input_name = sess.get_inputs()[0].name
    # label_name = sess.get_outputs()[0].name

    target_size = (192, 192)

    if True:
        raw_frame = cv2.imread("../images/before/human_image.jpg")
        pre_shape = raw_frame.shape[0:2][::-1]
        frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGRA2RGB)
        frame = preprocess(frame, target_size)
        frame = np.transpose(frame, (0, 2, 3, 1))
        print(frame.shape)
        pred = rknn.inference(inputs=[frame.astype(np.float32)])
        pred = pred[0]
        print(pred.shape)

        # without argmax
        pred = np.argmax(pred, axis=1)
        pred = pred[0]
        print(pred.shape)
        print("pred =")
        print(pred)
        raw_frame = resize(raw_frame, target_size)
        image = display_masked_image(pred, raw_frame)
        image = resize(image, target_size=pre_shape)
        cv2.imwrite('../images/after/RKNN_PC_Result.jpg', image)
