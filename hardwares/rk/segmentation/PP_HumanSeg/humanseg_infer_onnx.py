from utils.humanseg_tool import resize, preprocess, display_masked_image
import argparse
import cv2
import numpy as np
import os


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--model_path',
        dest='model_path',
        default="./weights/onnx/portrait_pp_humansegv2_lite_256x144_inference_model_with_softmax.onnx",
        type=str,
        help="path of model")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold of score.")
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


class Humanseg:
    def __init__(self, model_path,
                 target_size=None,
                 backend_type="onnx",
                 target="RK3588"):
        if target_size is None:
            target_size = (192, 192)
        self.model_path = model_path
        self.target = target
        self.target_size = target_size
        self.backend_type = backend_type

    def get_input_img(self, img):
        if isinstance(img, str):
            raw_frame = cv2.imread(img)
        else:
            raw_frame = img
        frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGRA2RGB)
        frame = preprocess(frame, self.target_size)
        return frame, raw_frame

    def infer_by_onnx(self, img):
        from utils.ONNXConfig import ONNXConfig
        frame, raw_frame = self.get_input_img(img)
        model = ONNXConfig(self.model_path)
        frame = np.array((frame,))
        pred = model.infer(frame.astype(np.float32))[0]
        return frame, raw_frame, pred

    def infer_by_rknn_pc(self, img, verbose=True):
        from utils.RKNNConfig import RKNNConfigPC
        frame, raw_frame = self.get_input_img(img)
        new_inputs = frame.transpose((1, 2, 0))  # chw hwc
        new_inputs = np.array((new_inputs,)).astype('float32')
        inputs = np.array((frame,))
        model = RKNNConfigPC(self.model_path, self.target).create_rknn(verbose=verbose)
        result = model.inference([new_inputs])
        return inputs, raw_frame, result

    def infer_by_rknn_board(self, img, verbose=True):
        from utils.RKNNConfig import RKNNConfigBoard
        frame, raw_frame = self.get_input_img(img)
        new_inputs = frame.transpose((1, 2, 0))  # chw hwc
        new_inputs = np.array((new_inputs,)).astype('float32')
        inputs = np.array((frame,))
        model = RKNNConfigBoard(self.model_path, self.target).create_rknn(verbose=verbose)
        result = model.inference([new_inputs])
        return inputs, raw_frame, result

    def predict(self, img):
        if self.backend_type == "onnx":
            frame, raw_frame, pred = self.infer_by_onnx(img)
            pred = pred[0]
        elif self.backend_type == "rk_pc":
            frame, raw_frame, pred = self.infer_by_rknn_pc(img, verbose=False)
            pred = pred[0][0]
        elif self.backend_type == "rk_board":
            frame, raw_frame, pred = self.infer_by_rknn_board(img, verbose=False)
            pred = pred[0][0]

        raw_frame = resize(raw_frame, self.target_size)
        pred = np.argmax(pred, axis=0)
        # print("pred.shape:{}".format(pred.shape))
        # print("raw_frame.shape:{}".format(raw_frame.shape))
        image = display_masked_image(pred, raw_frame)
        image = resize(image, target_size=raw_frame.shape[0:2][::-1])
        t_ls = os.path.basename(FLAGS.save_path).split(".")
        save_path = os.path.dirname(FLAGS.save_path) + "/" + t_ls[0] + "_" + self.backend_type + "." + t_ls[-1]
        cv2.imwrite(save_path, image)


if __name__ == "__main__":
    FLAGS = parse_args()
    humanseg = Humanseg(model_path=FLAGS.model_path, backend_type=FLAGS.backend_type)
    humanseg.predict(FLAGS.image_path)
    # pred = sess.run(
    #     [label_name],
    #     {input_name: frame.astype(np.float32)}
    # )[0]
    # pred = pred[0]
    # print(np.array(pred).shape)
    # raw_frame = resize(raw_frame, target_size)
    # image = display_masked_image(pred, raw_frame)
    # image = resize(image, target_size=pre_shape)
    # cv2.imwrite('../images/after/ONNXResult.jpg', image)
