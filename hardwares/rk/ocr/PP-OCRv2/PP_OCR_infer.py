import argparse
import cv2
import utils.predict_rec as predict_rec
import utils.predict_det as predict_det
import utils.predict_cls as predict_cls
from utils.predict_det import preprocess_boxes


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

    # params for rec classifier
    parser.add_argument("--rec_algorithm", type=str, default='CRNN')
    parser.add_argument("--rec_model_dir", type=str, default="./weights/onnx/PP_OCR_v2_rec.onnx")
    parser.add_argument("--rec_image_shape", type=str, default="3, 32, 960")
    parser.add_argument("--rec_batch_num", type=int, default=6)
    parser.add_argument("--max_text_length", type=int, default=25)
    parser.add_argument(
        "--rec_char_dict_path",
        type=str,
        default="./utils/doc/ppocr_keys_v1.txt")
    parser.add_argument("--use_space_char", type=str2bool, default=True)
    parser.add_argument(
        "--vis_font_path", type=str, default="./utils/doc/fonts/simfang.ttf")
    parser.add_argument("--drop_score", type=float, default=0.85)

    # params for text classifier
    parser.add_argument("--use_angle_cls", type=str2bool, default=True)
    parser.add_argument("--cls_model_dir", type=str, default="./weights/onnx/PP_OCR_v2_cls.onnx")
    parser.add_argument("--cls_image_shape", type=str, default="3, 32, 960")
    parser.add_argument("--label_list", type=list, default=['0', '180'])
    parser.add_argument("--cls_batch_num", type=int, default=6)
    parser.add_argument("--cls_thresh", type=float, default=0.9)

    parser.add_argument(
        '--backend_type',
        dest='backend_type',
        help='The type for reading the model.',
        type=str,
        choices=["rk_board", "rk_pc", "onnx"],
        default='rk_pc')
    parser.add_argument(
        '--target',
        dest='target',
        help='The target for build rknn.',
        type=str,
        default='RK3588')
    return parser.parse_args()


def postprocess(dt_boxes, rec_res):
    filter_boxes, filter_rec_res = [], []
    for box, rec_result in zip(dt_boxes, rec_res):
        text, score = rec_result
        if score >= FLAGS.drop_score:
            filter_boxes.append(box)
            filter_rec_res.append(rec_result)

    return filter_boxes, filter_rec_res


class PPOCR:
    def __init__(self, arg):
        self.arg = arg

    def predict(self, img_path):
        img = cv2.imread(img_path)
        img = cv2.resize(img, (960, 960))
        ori_im = img.copy()

        # text detect
        text_detector = predict_det.TextDetector(FLAGS)
        dt_boxes = text_detector(img)
        dt_boxes, img_crop_list = preprocess_boxes(dt_boxes, ori_im)
        tmp = text_detector.draw_det(ori_im, dt_boxes)
        cv2.imwrite("./images/after/temp.jpg",tmp)

        # text classifier
        if FLAGS.use_angle_cls:
            text_classifier = predict_cls.TextClassifier(FLAGS)
            img_crop_list, angle_list = text_classifier(img_crop_list)
        print(angle_list)
        # text recognize
        # text_recognizer = predict_rec.TextRecognizer(FLAGS)
        # rec_res = text_recognizer(img_crop_list)
        # _, filter_rec_res = postprocess(dt_boxes, rec_res)
        #
        # for text, score in filter_rec_res:
        #     print("{}, {:.3f}".format(text, score))
        # print("Finish!")


if __name__ == "__main__":
    FLAGS = parse_args()
    pp_ocr = PPOCR(FLAGS)
    pp_ocr.predict(FLAGS.image_path)
