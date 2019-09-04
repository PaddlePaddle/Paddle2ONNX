# Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.
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
import cv2
import numpy as np
from PIL import Image

DATA_DIM = 224
img_mean = np.array([0.485, 0.456, 0.406]).reshape((1, 1, 3))
img_std = np.array([0.229, 0.224, 0.225]).reshape((1, 1, 3))


class SSDReader():
    """
    The reader for testing image detection model, reader the voc data
    """

    def __init__(self, image_path):
        mean_value = [127.5, 127.5, 127.5]
        self.image_path = image_path
        self._resize_height = 300
        self._resize_width = 300
        self._img_mean = np.array(mean_value)[:, np.newaxis, np.newaxis].astype(
            'float32')

    def reader(self, program, feed_target_names):
        walk_list = os.walk(self.image_path)
        for path, list_dir, file_list in walk_list:
            for file_name in file_list:
                outputs = []
                file_path = os.path.join(path, file_name)
                img = Image.open(file_path)
                if img.mode == 'L':
                    img = im.convert('RGB')
                im_width, im_height = img.size
                img = img.resize((self._resize_width, self._resize_height),
                                 Image.ANTIALIAS)
                img = np.array(img)
                if len(img.shape) == 3:
                    img = np.swapaxes(img, 1, 2)
                    img = np.swapaxes(img, 1, 0)
                img = img[[2, 1, 0], :, :]
                img = img.astype('float32')
                img -= self._img_mean
                img = img * 0.007843
                img = np.expand_dims(img, axis=0)
                outputs.append(img)
                yield outputs


class YoloReader():
    """
    The reader for testing image detection model, reader the voc data
    """

    def __init__(self, image_path):
        self.image_path = image_path
        self._resize_height = 608
        self._resize_width = 608

    def reader(self, program, feed_target_names):
        walk_list = os.walk(self.image_path)
        for path, list_dir, file_list in walk_list:
            for file_name in file_list:
                outputs = []
                file_path = os.path.join(path, file_name)
                with open(file_path, 'rb') as f:
                    im = f.read()
                data = np.frombuffer(im, dtype='uint8')
                im = cv2.imdecode(data, 1)  # BGR mode, but need RGB mod
                img = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                im_height, im_width, _ = img.shape
                img = cv2.resize(
                    img, (self._resize_height, self._resize_width),
                    interpolation=2)
                img = np.array(img)
                img = img / 255.0
                img = img.astype('float32')
                img -= img_mean
                img = img / img_std
                if len(img.shape) == 3:
                    img = np.swapaxes(img, 1, 2)
                    img = np.swapaxes(img, 1, 0)
                img = np.expand_dims(img, axis=0)
                img_size = np.array([im_height, im_width]).reshape(
                    -1, 2).astype('int32')
                img = img.astype('float32')
                outputs.append(img)
                outputs.append(img_size)
                yield outputs
