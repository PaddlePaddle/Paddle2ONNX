import os
import sys

sys.path.append(os.getcwd())

import argparse
from utils.bisenet_tool import *


def parse_args():
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument(
        '--model_path',
        dest='model_path',
        default="./weights/onnx/bisenet.onnx",
        type=str,
        help="path of model")
    parser.add_argument(
        '--image_path',
        dest='image_path',
        default="./images/before/bisenet_demo_input.jpeg",
        type=str,
        help='The directory or path or file list of the images to be predicted.')
    parser.add_argument(
        '--backend_type',
        dest='backend_type',
        help='The type for reading the model.',
        type=str,
        choices=["rk_board", "rk_pc", "onnx"],
        default='rk_pc')
    parser.add_argument(
        '--save_path',
        dest='save_path',
        help='The file for saving the predict result.',
        type=str,
        default='./images/after/bisenet_demo_output.png')
    parser.add_argument(
        '--target',
        dest='target',
        help='The target for build rknn.',
        type=str,
        default='RK3588')
    return parser.parse_args()


class Bisenet:
    def __init__(self, model_path,
                 target_size=None,
                 backend_type="onnx",
                 target="RK3588"):
        if target_size is None:
            target_size = (1024, 1024)
        self.model_path = model_path
        self.target = target
        self.target_size = target_size
        self.backend_type = backend_type

    def infer_by_onnx(self, img):
        from utils.ONNXConfig import ONNXConfig
        model = ONNXConfig(self.model_path)
        # model.do_bisenet_change()
        results = model.infer(img)
        return results

    def infer_by_rknn_pc(self, img, verbose=True):
        from utils.RKNNConfig import RKNNConfigPC
        model = RKNNConfigPC(self.model_path, self.target).create_rknn(verbose=verbose,outputs=['bilinear_interp_v2_2.tmp_0'])
        result = model.inference([img])
        # result = np.argmax(result,axis=1)
        return result

    def infer_by_rknn_board(self, img, verbose=False):
        from utils.RKNNConfig import RKNNConfigBoard
        model = RKNNConfigBoard(self.model_path, self.target).create_rknn(verbose=verbose)
        result = model.inference([img])
        return result

    def predict(self, img):
        if self.backend_type == "onnx":
            data = preprocess(img, self.target_size)
            data = np.expand_dims(data, axis=0)
            print(data.reshape((1024,1024,3)))
            pred = self.infer_by_onnx(data)
            pred = pred[0]
            # pred = np.argmax(pred, axis=1)
            print(pred)
        elif self.backend_type == "rk_pc":
            data = preprocess(img, self.target_size,self.backend_type == "onnx")
            data = np.expand_dims(data, axis=0)
            pred = self.infer_by_rknn_pc(data, verbose=False)
            pred = pred[0]
            print(pred.shape)
        elif self.backend_type == "rk_board":
            data = preprocess(img, self.target_size,self.backend_type == "onnx")
            data = np.expand_dims(data, axis=0)
            pred = self.infer_by_rknn_board(data, verbose=False)
            pred = pred[0]
        pred = np.argmax(pred, axis=1)
        print(pred)
        save_imgs(pred, FLAGS.backend_type, FLAGS.save_path)


if __name__ == '__main__':
    FLAGS = parse_args()
    bisenet = Bisenet(FLAGS.model_path,
                      backend_type=FLAGS.backend_type,
                      target=FLAGS.target)
    bisenet.predict(FLAGS.image_path)
