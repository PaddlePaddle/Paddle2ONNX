# Copyright (c) 2021  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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

import paddle
from onnxbase import APIOnnx
from onnxbase import randtool


class Net(paddle.nn.Layer):
    """
    simple Net
    """

    def __init__(self):
        super(Net, self).__init__()

    def forward(self, bboxes, scores):
        """
        forward
        """
        N = 1
        M = 1200
        C = 21
        BOX_SIZE = 4
        background = 0
        nms_threshold = 0.3
        nms_top_k = 400
        keep_top_k = 200
        score_threshold = 0.01
        x = paddle.fluid.layers.multiclass_nms(
            bboxes,
            scores,
            score_threshold,
            nms_top_k,
            keep_top_k,
            nms_threshold=nms_threshold,
            normalized=True,
            nms_eta=1.0,
            background_label=0)
        return x


def test_multiclass_nms():
    """
    api: paddle.divide
    op version: 9
    """
    op = Net()
    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, 'multiclass_nms', [12])
    import numpy as np

    def init_test_input():
        def softmax(x):
            # clip to shiftx, otherwise, when calc loss with
            # log(exp(shiftx)), may get log(0)=INF
            shiftx = (x - np.max(x)).clip(-64.)
            exps = np.exp(shiftx)
            return exps / np.sum(exps)

        N = 1
        M = 1200
        C = 21
        BOX_SIZE = 4
        background = 0
        nms_threshold = 0.3
        nms_top_k = 400
        keep_top_k = 200
        score_threshold = 0.01

        scores = np.random.random((N * M, C)).astype('float32')

        scores = np.apply_along_axis(softmax, 1, scores)
        scores = np.reshape(scores, (N, M, C))
        scores = np.transpose(scores, (0, 2, 1))

        boxes = np.random.random((N, M, BOX_SIZE)).astype('float32')
        boxes[:, :, 0:2] = boxes[:, :, 0:2] * 0.5
        boxes[:, :, 2:4] = boxes[:, :, 2:4] * 0.5 + 0.5
        return boxes, scores

    boxes, scores = init_test_input()
    bboxes = paddle.to_tensor(boxes, "float32")
    scores = paddle.to_tensor(scores, "float32")
    obj.set_input_data("input_data", bboxes, scores)
    obj.run()
