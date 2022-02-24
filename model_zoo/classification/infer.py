# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys

sys.path.insert(0, ".")
import argparse
import numpy as np
from PIL import Image

import paddle
from onnxruntime import InferenceSession

# 从模型代码中导入模型
from utils.presets import ClassificationPresetEval
from utils.presets import Topk


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--model_path',
        type=str,
        default="/Users/huangshenghui/PP/Paddle2ONNX/model_zoo/classification/ResNet50_vd_infer/inference",
        help="paddle model filename")
    parser.add_argument(
        '--onnx_file',
        type=str,
        default="/Users/huangshenghui/PP/Paddle2ONNX/model_zoo/classification/ResNet50_vd_infer/inference.onnx",
        help="onnx model filename")
    parser.add_argument(
        '--img_path',
        type=str,
        default="./images/ILSVRC2012_val_00000010.jpeg",
        help="image filename")
    parser.add_argument('--crop_size', default=224, help='crop_szie')
    parser.add_argument('--resize_size', default=256, help='resize_size')
    return parser.parse_args()


def preprocess(img_path):
    # define transforms
    eval_transforms = ClassificationPresetEval(
        crop_size=FLAGS.crop_size, resize_size=FLAGS.resize_size)
    # 准备输入
    with open(img_path, 'rb') as f:
        img = Image.open(f).convert('RGB')

    data = eval_transforms(img)
    data = np.expand_dims(data, axis=0)
    return data


def postprocess(results):
    topk = Topk(5, "./utils/imagenet1k_label_list.txt")
    data = topk(results)
    # class_id = results.argmax()
    # prob = results[0][class_id]
    return data


def print_detail(batch_results):
    for number, result_dict in enumerate(batch_results):
        clas_ids = result_dict["class_ids"]
        scores_str = "[{}]".format(", ".join("{:.2f}".format(r)
                                             for r in result_dict["scores"]))
        label_names = result_dict["label_names"]
        print("class id(s): {}, score(s): {}, label_name(s): {}".format(
            clas_ids, scores_str, label_names))


def paddle_predict(model_path, imgs_path):
    model = paddle.jit.load(model_path)
    model.eval()
    data = preprocess(imgs_path)
    results = model(data).numpy()
    results = postprocess(results)
    return results


def onnx_predict(onnx_path, imgs_path):
    sess = InferenceSession(onnx_path)
    data = preprocess(imgs_path)
    results = sess.run(None, input_feed={sess.get_inputs()[0].name: data})[0]
    results = postprocess(results)
    return results


if __name__ == '__main__':
    FLAGS = parse_args()

    data = paddle_predict(FLAGS.model_path, FLAGS.img_path)
    print_detail(data)
    # print(f"paddle result: class_id: {class_id}, prob: {prob}")

    data = onnx_predict(FLAGS.onnx_file, FLAGS.img_path)
    print_detail(data)
    # print(f"onxx result: class_id: {class_id}, prob: {prob}")
