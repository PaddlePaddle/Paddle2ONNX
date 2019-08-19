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

import numpy as np


def image_classification_random_reader(program, feed_target_names):
    """
    Get the random reader to feed image classification inputs.
    """
    a = 0.0
    b = 1.0
    input_shapes = [
        program.global_block().var(var_name).shape
        for var_name in feed_target_names
    ]
    input_shapes = [
        shape if shape[0] > 0 else (1, ) + shape[1:] for shape in input_shapes
    ]
    for i in range(1):
        # Generate dummy data as inputs
        inputs = [(b - a) * np.random.random(shape).astype("float32") + a
                  for shape in input_shapes]
        yield inputs
