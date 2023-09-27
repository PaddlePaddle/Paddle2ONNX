import unittest

import onnxruntime as rt
import numpy as np
import paddle.inference as paddle_infer
import cv2
from paddle.inference import PrecisionType, PlaceType
from paddle.inference import convert_to_mixed_precision

def preprocess(image_path):
    """ Preprocess input image file
    Args:
        image_path(str): Path of input image file


    Returns:
        preprocessed data(np.ndarray): Shape of [N, C, H, W]
    """

    def resize_by_short(im, resize_size):
        short_size = min(im.shape[0], im.shape[1])
        scale = 256 / short_size
        new_w = int(round(im.shape[1] * scale))
        new_h = int(round(im.shape[0] * scale))
        return cv2.resize(im, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    def center_crop(im, crop_size):
        h, w, c = im.shape
        w_start = (w - crop_size) // 2
        h_start = (h - crop_size) // 2
        w_end = w_start + crop_size
        h_end = h_start + crop_size
        return im[h_start:h_end, w_start:w_end, :]

    def normalize(im, mean, std):
        im = im.astype("float32") / 255.0
        # to rgb
        im = im[:, :, ::-1]
        mean = np.array(mean).reshape((1, 1, 3)).astype("float32")
        std = np.array(std).reshape((1, 1, 3)).astype("float32")
        return (im - mean) / std

    # resize the short edge to `resize_size`
    im = cv2.imread(image_path)
    resized_im = resize_by_short(im, 256)

    # crop from center
    croped_im = center_crop(resized_im, 224)

    # normalize
    normalized_im = normalize(croped_im, [0.485, 0.456, 0.406],
                              [0.229, 0.224, 0.225])

    # transpose to NCHW
    data = np.expand_dims(normalized_im, axis=0)
    data = np.transpose(data, (0, 3, 1, 2))

    return data


def predict_by_onnx(input_data):
    sess = rt.InferenceSession('inference_fp16.onnx', providers=['CPUExecutionProvider'])
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    pred_onnx = sess.run([output_name], {input_name: input_data})
    return np.array(pred_onnx[0])


def predict_by_paddle_inference(input_data):
    config = paddle_infer.Config("inference.pdmodel", "inference.pdiparams")
    config.enable_use_gpu(500, 0)

    predictor = paddle_infer.create_predictor(config)
    # 获取输入的名称
    input_names = predictor.get_input_names()
    input_handle = predictor.get_input_handle(input_names[0])

    # 设置输入
    input_handle.copy_from_cpu(input_data)

    # 运行predictor
    predictor.run()

    # 获取输出
    output_names = predictor.get_output_names()
    output_handle = predictor.get_output_handle(output_names[0])
    output_data = output_handle.copy_to_cpu()  # numpy.ndarray类型
    return output_data


class TestCalculator(unittest.TestCase):
    def creat_fp16_model(src_model,src_params):
        black_list = set()
        dst_model = "./inference_fp16.pdmodel"
        dst_params = "./inference_fp16.pdiparams"

        convert_to_mixed_precision(
            src_model,  # fp32模型文件路径
            src_params,  # fp32权重文件路径
            dst_model,  # 混合精度模型文件保存路径
            dst_params,  # 混合精度权重文件保存路径
            PrecisionType.Half,  # 转换精度，如 PrecisionType.Half
            PlaceType.GPU,  # 后端，如 PlaceType.GPU
            True,  # 保留输入输出精度信息，若为 True 则输入输出保留为 fp32 类型，否则转为 precision 类型
            black_list  # 黑名单列表，哪些 op 不需要进行精度类型转换
        )

    def testDivide01(self):
        cal = Calculator()
        result = cal.divide(10, 2)
        self.assertEqual(result, 5)

    def testDivide02(self):
        cal = Calculator()
        result = cal.divide(20, 0.5)
        self.assertEqual(result, 40)


# if __name__ == "__main__":
#     input_data = preprocess("ILSVRC2012_val_00000010.jpeg")

#     data_onnx = predict_by_onnx(input_data)
#     data_paddle_inference = predict_by_paddle_inference(input_data)

#     sim_0 = np.fabs(data_onnx - data_paddle_inference)
#     print(np.max(sim_0), np.min(sim_0), np.mean(sim_0))
#     sim_1 = np.fabs(data_onnx - data_paddle_inference) / (np.fabs(data_paddle_inference) + 1e-06)
#     print(np.max(sim_1), np.min(sim_1), np.mean(sim_1))

#     print(np.fabs(data_paddle_inference.max()), np.fabs(data_paddle_inference.min()))

if __name__ == '__main__':
    unittest.main()
