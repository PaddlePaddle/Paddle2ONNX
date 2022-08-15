import argparse
import cv2
from pp_ocr_utils.predict_det import DetPreProcess, Det, det_mean, det_std
import sys
import numpy as np

sys.path.append('../../')


def str2bool(v):
    return v.lower() in ("true", "t", "1")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    # params for text detector
    parser.add_argument("--image_path", type=str, default="./images/before/lite_demo_input.png")
    parser.add_argument("--det_algorithm", type=str, default='DB')
    parser.add_argument("--det_model_dir", type=str, default="./weights/onnx/PP_OCR_v2_det.onnx")
    parser.add_argument("--det_limit_side_len", type=float, default=960)
    parser.add_argument("--det_limit_type", type=str, default='max')

    # DB parmas
    parser.add_argument("--det_db_thresh", type=float, default=0.6)
    parser.add_argument("--det_db_box_thresh", type=float, default=0.7)
    parser.add_argument("--det_db_unclip_ratio", type=float, default=1.5)
    parser.add_argument("--use_dilation", type=str2bool, default=False)
    parser.add_argument("--det_db_score_mode", type=str, default="fast")

    parser.add_argument(
        '--backend_type',
        dest='backend_type',
        help='The type for reading the model.',
        type=str,
        choices=["rk_board", "rk_pc", "onnx"],
        default='onnx')
    parser.add_argument(
        '--target',
        dest='target',
        help='The target for build rknn.',
        type=str,
        default='RK3588')
    return parser.parse_args()


if __name__ == "__main__":
    FLAGS = parse_args()

    det_preprocess = DetPreProcess()
    ori_im, src_img = det_preprocess.get_inputs(FLAGS.image_path, do_normalize=FLAGS.backend_type == "onnx")
    ori_im = np.expand_dims(ori_im, axis=0)

    # ocr det
    if FLAGS.backend_type == "onnx":
        from utils.onnx_config import ONNXConfig

        model = ONNXConfig(FLAGS.det_model_dir, need_show=True)
    elif FLAGS.backend_type == "rk_pc":
        from utils.rknn_config import RKNNConfigPC

        # config
        rknn_std = [[round(std * 255, 3) for std in det_std]]
        rknn_mean = [[round(mean * 255, 3) for mean in det_mean]]
        # read model
        model = RKNNConfigPC(model_path=FLAGS.det_model_dir,
                             mean_values=rknn_mean,
                             std_values=rknn_std)
    elif FLAGS.backend_type == "rk_board":
        from utils.rknn_config import RKNNConfigBoard

        # read model
        model = RKNNConfigBoard(rknn_path=FLAGS.det_model_dir)

    result, use_time = model.infer(ori_im)
    det = Det(thresh=FLAGS.det_db_thresh,
              box_thresh=FLAGS.det_db_box_thresh,
              max_candidates=1000,
              unclip_ratio=FLAGS.det_db_unclip_ratio,
              use_dilation=FLAGS.use_dilation,
              score_mode=FLAGS.det_db_score_mode)
    dt_boxes = det.predict(result, src_img, "./images/after/result.jpg")
