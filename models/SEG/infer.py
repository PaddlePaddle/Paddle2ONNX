# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import sys

LOCAL_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(LOCAL_PATH, '..', '..'))

import numpy as np

import paddleseg.transforms as T
from paddleseg.cvlibs import manager
from paddleseg.utils import logger, get_image_list
from paddleseg.utils.visualize import get_pseudo_color_map

import onnxruntime as rt
import paddle


def parse_args():
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument(
        '--model_path',
        dest='model_path',
        help='The path of the paddle pdmodel',
        type=str,
        default='fcn/model')
    parser.add_argument(
        '--onnx_path',
        dest='onnx_path',
        help='file of onnx of model.',
        type=str,
        default='/Users/huangshenghui/PP/Paddle2ONNX/paddle2onnx/model.onnx')
    parser.add_argument(
        '--image_path',
        dest='image_path',
        help='The directory or path or file list of the images to be predicted.',
        type=str,
        default='images/cityscapes_demo.png')
    parser.add_argument(
        '--batch_size',
        dest='batch_size',
        help='Mini batch size of one gpu or cpu.',
        type=int,
        default=1)
    parser.add_argument(
        '--save_dir',
        dest='save_dir',
        help='The directory for saving the predict result.',
        type=str,
        default='./outputs')
    parser.add_argument(
        '--with_argmax',
        dest='with_argmax',
        help='Perform argmax operation on the predict result.',
        action='store_true')
    return parser.parse_args()


class Predictor:
    def __init__(self, args):
        t = [{'type': 'Normalize'}]
        self._transforms = self.load_transforms(t)
        self.args = args

    @property
    def transforms(self):
        return self._transforms

    @staticmethod
    def load_transforms(t_list):
        com = manager.TRANSFORMS
        transforms = []
        for t in t_list:
            ctype = t.pop('type')
            transforms.append(com[ctype](**t))

        return T.Compose(transforms)

    def run(self, imgs_path):
        if not isinstance(imgs_path, (list, tuple)):
            imgs_path = [imgs_path]

        args = self.args
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)

        for i in range(0, len(imgs_path), args.batch_size):
            data = np.array([
                self._preprocess(p) for p in imgs_path[i:i + args.batch_size]
            ])

            # run paddle inference
            model = paddle.jit.load(args.model_path)
            model.eval()
            paddle_outs = model(data)
            results = paddle_outs.numpy()
            self._save_imgs(results, imgs_path[i:i + args.batch_size], "paddle")

            # run onnxruntime
            sess = rt.InferenceSession(args.onnx_path)
            input_name = sess.get_inputs()[0].name
            label_name = sess.get_outputs()[0].name
            print("sess input/output name : ", input_name, label_name)
            ort_outs = sess.run(None, {input_name: data})

            diff = ort_outs[0] - results
            max_abs_diff = np.fabs(diff).max()
            if max_abs_diff < 1e-05:
                print(
                    "The difference of results between ONNXRuntime and Paddle looks good!"
                )
            else:
                relative_diff = max_abs_diff / np.fabs(results).max()
                if relative_diff < 1e-05:
                    print(
                        "The difference of results between ONNXRuntime and Paddle looks good!"
                    )
                else:
                    print(
                        "The difference of results between ONNXRuntime and Paddle looks bad!"
                    )
                print('relative_diff: ', relative_diff)
            print('max_abs_diff: ', max_abs_diff)
            self._save_imgs(results, imgs_path[i:i + args.batch_size], "onnx")

        logger.info("Finish")

    def _preprocess(self, img):
        return self.transforms(img)[0]

    def _postprocess(self, results):
        if self.args.with_argmax:
            results = np.argmax(results, axis=1)
        return results

    def _save_imgs(self, results, imgs_path, prefix=None):
        for i in range(results.shape[0]):
            result = get_pseudo_color_map(results[i])
            basename = os.path.basename(imgs_path[i])
            basename, _ = os.path.splitext(basename)
            if prefix is not None and isinstance(prefix, str):
                basename = prefix + "_" + basename
            basename = f'{basename}.png'
            result.save(os.path.join(self.args.save_dir, basename))


def main(args):
    imgs_list, _ = get_image_list(args.image_path)

    predictor = Predictor(args)
    predictor.run(imgs_list)


if __name__ == '__main__':
    args = parse_args()
    main(args)
