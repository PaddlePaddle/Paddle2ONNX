# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
from onnxbase import _test_only_pir


class BaseNet(paddle.nn.Layer):
    def __init__(self, config):
        super(BaseNet, self).__init__()
        self.score_threshold = config["score_threshold"]
        self.nms_top_k = config["nms_top_k"]
        self.keep_top_k = config["keep_top_k"]
        self.nms_threshold = config["nms_threshold"]
        self.normalized = config["normalized"]
        self.nms_eta = config["nms_eta"]
        self.background_label = config["background_label"]

    def forward(self, bboxes, scores):
        out, index, nms_rois_num = paddle._C_ops.multiclass_nms3(
            bboxes,
            scores,
            None,
            self.score_threshold,
            self.nms_top_k,
            self.keep_top_k,
            self.nms_threshold,
            self.normalized,
            self.nms_eta,
            self.background_label,
        )
        return out, index, nms_rois_num


@_test_only_pir
def test_multiclass_nms3_1():
    config = {
        "score_threshold": 0.001,
        "nms_top_k": 1000,
        "keep_top_k": 10,
        "nms_threshold": 0.65,
        "normalized": True,
        "nms_eta": 1.0,
        "background_label": -1,
    }
    op = BaseNet(config)
    op.eval()
    obj = APIOnnx(op, "multiclass_nms3", [11])
    bboxes = paddle.randn([1, 3549, 4], dtype="float32")
    scores = paddle.randn([1, 80, 3549], dtype="float32")
    obj.set_input_data("input_data", (bboxes, scores))
    obj.run()


@_test_only_pir
def test_multiclass_nms3_2():
    config = {
        "score_threshold": 0.01,
        "nms_top_k": 100,
        "keep_top_k": 10,
        "nms_threshold": 0.7,
        "normalized": True,
        "nms_eta": 1.0,
        "background_label": 80,
    }
    op = BaseNet(config)
    op.eval()
    obj = APIOnnx(op, "multiclass_nms3", [11])
    bboxes = paddle.randn([1, 8400, 4], dtype="float32")
    scores = paddle.randn([1, 80, 8400], dtype="float32")
    obj.set_input_data("input_data", (bboxes, scores))
    obj.run()


if __name__ == "__main__":
    test_multiclass_nms3_1()
    test_multiclass_nms3_2()
