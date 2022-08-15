import argparse
from pp_humanseg_utils.pp_humanseg import HumanSegPreProcess, humanseg_std, humanseg_mean, Humanseg
import numpy as np
import sys

sys.path.append('../../')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--model_path',
        dest='model_path',
        default="./weights/onnx/portrait_pp_humansegv2_lite_256x144_inference_model_with_softmax.onnx",
        type=str,
        help="path of model")
    parser.add_argument(
        '--image_path',
        dest='image_path',
        default="./images/before/PP_HumanSeg_demo_input.jpg",
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
        default='./images/after/PP_HumanSeg_demo_output.png')
    parser.add_argument(
        '--target',
        dest='target',
        help='The target for build rknn.',
        type=str,
        default='RK3588')
    return parser.parse_args()


if __name__ == "__main__":
    FLAG = parse_args()
    human_seg_preprocess = HumanSegPreProcess()
    inputs, src_image = human_seg_preprocess.get_inputs(FLAG.image_path, do_normalize=(FLAG.backend_type == "onnx"))
    new_inputs = np.array((inputs,)).astype('float32')
    inputs = np.array((inputs,))

    # read model
    if FLAG.backend_type == "onnx":
        from utils.onnx_config import ONNXConfig

        model = ONNXConfig(FLAG.model_path,need_show=True)
    elif FLAG.backend_type == "rk_pc":
        from utils.rknn_config import RKNNConfigPC

        # config
        rknn_std = [[round(std * 255, 3) for std in humanseg_std]]
        rknn_mean = [[round(mean * 255, 3) for mean in humanseg_mean]]
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
    print(new_inputs.shape)
    result, use_time = model.infer(new_inputs)
    print("推理时间为{}ms".format(round(use_time * 1000, 2)))

    humanseg = Humanseg()
    humanseg.predict(results=result,
                     src_image=src_image,
                     save_path=FLAG.save_path,
                     backend_type=FLAG.backend_type)
