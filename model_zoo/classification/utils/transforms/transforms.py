#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import math
import numbers
import random
import warnings
import numpy as np
from PIL import Image
from enum import Enum

from collections.abc import Sequence
from typing import List, Tuple, Any, Optional

import paddle
from paddle import Tensor

try:
    import accimage
except ImportError:
    accimage = None


class InterpolationMode(Enum):
    """Interpolation modes
    Available interpolation methods are ``nearest``, ``bilinear``, ``bicubic``, ``box``, ``hamming``, and ``lanczos``.
    """
    NEAREST = "nearest"
    BILINEAR = "bilinear"
    BICUBIC = "bicubic"
    # For PIL compatibility
    BOX = "box"
    HAMMING = "hamming"
    LANCZOS = "lanczos"


def _interpolation_modes_from_int(i: int) -> InterpolationMode:
    inverse_modes_mapping = {
        0: InterpolationMode.NEAREST,
        2: InterpolationMode.BILINEAR,
        3: InterpolationMode.BICUBIC,
        4: InterpolationMode.BOX,
        5: InterpolationMode.HAMMING,
        1: InterpolationMode.LANCZOS,
    }
    return inverse_modes_mapping[i]


pil_modes_mapping = {
    InterpolationMode.NEAREST: 0,
    InterpolationMode.BILINEAR: 2,
    InterpolationMode.BICUBIC: 3,
    InterpolationMode.BOX: 4,
    InterpolationMode.HAMMING: 5,
    InterpolationMode.LANCZOS: 1,
}


def _is_pil_image(img: Any) -> bool:
    if accimage is not None:
        return isinstance(img, (Image.Image, accimage.Image))
    else:
        return isinstance(img, Image.Image)


def _get_image_size(img: Any) -> List[int]:
    if _is_pil_image(img):
        return img.size
    raise TypeError("Unexpected type {}".format(type(img)))


def _setup_size(size, error_msg):
    if isinstance(size, numbers.Number):
        return int(size), int(size)

    if isinstance(size, Sequence) and len(size) == 1:
        return size[0], size[0]

    if len(size) != 2:
        raise ValueError(error_msg)

    return size


def crop(img: Image.Image, top: int, left: int, height: int,
         width: int) -> Image.Image:
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    return img.crop((left, top, left + width, top + height))


def resize_pil(img, size, interpolation=Image.BILINEAR, max_size=None):
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))
    if not (isinstance(size, int) or
            (isinstance(size, Sequence) and len(size) in (1, 2))):
        raise TypeError('Got inappropriate size arg: {}'.format(size))

    if isinstance(size, Sequence) and len(size) == 1:
        size = size[0]
    if isinstance(size, int):
        w, h = img.size

        short, long = (w, h) if w <= h else (h, w)
        if short == size:
            return img

        new_short, new_long = size, int(size * long / short)

        if max_size is not None:
            if max_size <= size:
                raise ValueError(
                    f"max_size = {max_size} must be strictly greater than the requested "
                    f"size for the smaller edge size = {size}")
            if new_long > max_size:
                new_short, new_long = int(max_size * new_short /
                                          new_long), max_size

        new_w, new_h = (new_short, new_long) if w <= h else (new_long,
                                                             new_short)
        return img.resize((new_w, new_h), interpolation)
    else:
        if max_size is not None:
            raise ValueError(
                "max_size should only be passed if size specifies the length of the smaller edge, "
                "i.e. size should be an int or a sequence of length 1 in deploy mode."
            )
        return img.resize(size[::-1], interpolation)


def resize(img: Tensor,
           size: List[int],
           interpolation: InterpolationMode=InterpolationMode.BILINEAR,
           max_size: Optional[int]=None,
           antialias: Optional[bool]=None) -> Tensor:
    # Backward compatibility with integer value
    if isinstance(interpolation, int):
        warnings.warn(
            "Argument interpolation should be of type InterpolationMode instead of int. "
            "Please, use InterpolationMode enum.")
        interpolation = _interpolation_modes_from_int(interpolation)

    if not isinstance(interpolation, InterpolationMode):
        raise TypeError("Argument interpolation should be a InterpolationMode")

    if not isinstance(img, paddle.Tensor):
        if antialias is not None and not antialias:
            warnings.warn(
                "Anti-alias option is always applied for PIL Image input. Argument antialias is ignored."
            )
        pil_interpolation = pil_modes_mapping[interpolation]
        return resize_pil(
            img, size=size, interpolation=pil_interpolation, max_size=max_size)


def center_crop(img: Tensor, output_size: List[int]) -> Tensor:
    if isinstance(output_size, numbers.Number):
        output_size = (int(output_size), int(output_size))
    elif isinstance(output_size, (tuple, list)) and len(output_size) == 1:
        output_size = (output_size[0], output_size[0])

    image_width, image_height = _get_image_size(img)
    crop_height, crop_width = output_size

    if crop_width > image_width or crop_height > image_height:
        padding_ltrb = [
            (crop_width - image_width) // 2 if crop_width > image_width else 0,
            (crop_height - image_height) // 2
            if crop_height > image_height else 0,
            (crop_width - image_width + 1) // 2
            if crop_width > image_width else 0,
            (crop_height - image_height + 1) // 2
            if crop_height > image_height else 0,
        ]
        img = pad(img, padding_ltrb, fill=0)  # PIL uses fill value 0
        image_width, image_height = _get_image_size(img)
        if crop_width == image_width and crop_height == image_height:
            return img

    crop_top = int(round((image_height - crop_height) / 2.))
    crop_left = int(round((image_width - crop_width) / 2.))
    return crop(img, crop_top, crop_left, crop_height, crop_width)


__all__ = [
    "Compose",
    "Resize",
    "CenterCrop",
]


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img


class Resize(paddle.nn.Layer):
    def __init__(self,
                 size,
                 interpolation=InterpolationMode.BILINEAR,
                 max_size=None,
                 antialias=None):
        super().__init__()
        if not isinstance(size, (int, Sequence)):
            raise TypeError("Size should be int or sequence. Got {}".format(
                type(size)))
        if isinstance(size, Sequence) and len(size) not in (1, 2):
            raise ValueError(
                "If size is a sequence, it should have 1 or 2 values")
        self.size = size
        self.max_size = max_size

        # Backward compatibility with integer value
        if isinstance(interpolation, int):
            warnings.warn(
                "Argument interpolation should be of type InterpolationMode instead of int. "
                "Please, use InterpolationMode enum.")
            interpolation = _interpolation_modes_from_int(interpolation)

        self.interpolation = interpolation
        self.antialias = antialias

    def forward(self, img):
        return resize(img, self.size, self.interpolation, self.max_size,
                      self.antialias)


class CenterCrop(paddle.nn.Layer):
    def __init__(self, size):
        super().__init__()
        self.size = _setup_size(
            size,
            error_msg="Please provide only two dimensions (h, w) for size.")

    def forward(self, img):
        return center_crop(img, self.size)
