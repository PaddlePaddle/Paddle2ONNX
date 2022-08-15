import argparse
import cv2
import numpy as np
import sys
from picodet_utils.picodet import PicodetPreProcess, Picodet, picodet_std, picodet_mean

sys.path.append('../../')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--model_path',
        dest='model_path',
        default="./weights/onnx/picodet_s_320_coco_sim.onnx",
        type=str,
        help="path of model")
    parser.add_argument(
        '--image_path',
        dest='image_path',
        default="./images/before/picodet_demo_input.jpg",
        type=str,
        help='The directory or path or file list of the images to be predicted.')
    parser.add_argument(
        '--backend_type',
        dest='backend_type',
        help='The type for reading the model.',
        type=str,
        choices=["rk_board", "rk_pc", "onnx"],
        default='onnx')
    parser.add_argument(
        '--save_path',
        dest='save_path',
        help='The file for saving the predict result.',
        type=str,
        default='./images/after/picodet_demo_input.jpg')
    return parser.parse_args()


if __name__ == "__main__":
    FLAG = parse_args()

    # read img
    pic_pre_process = PicodetPreProcess()
    inputs, src_image = pic_pre_process.get_inputs(FLAG.image_path, do_normalize=(FLAG.backend_type == "onnx"))
    new_inputs = np.array((inputs,)).astype('float32')
    inputs = np.array((inputs,))


    # read model
    if FLAG.backend_type == "onnx":
        from utils.onnx_config import ONNXConfig

        model = ONNXConfig(FLAG.model_path)
    elif FLAG.backend_type == "rk_pc":
        from utils.rknn_config import RKNNConfigPC
        # config
        rknn_std = [[round(std * 255, 3) for std in picodet_std]]
        rknn_mean = [[round(mean * 255, 3) for mean in picodet_mean]]
        # print("std:{},mean:{}".format(rknn_std, rknn_mean))
        # read model
        model = RKNNConfigPC(model_path=FLAG.model_path,
                             mean_values=rknn_mean,
                             std_values=rknn_std)
    elif FLAG.backend_type == "rk_board":
        from utils.rknn_config import RKNNConfigBoard
        # read model
        model = RKNNConfigBoard(rknn_path=FLAG.model_path)

    # infer
    result, use_time = model.infer(new_inputs)
    print("推理时间为{}ms".format(round(use_time * 1000, 2)))

    # predict
    picodet = Picodet()
    picodet.predict(result=result,
                    src_image=src_image,
                    save_path=FLAG.save_path,
                    backend_type=FLAG.backend_type)
