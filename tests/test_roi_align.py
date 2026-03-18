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
    def __init__(self):
        super(BaseNet, self).__init__()
        self.output_size = 3
        self.spatial_scale = 1.0
        self.sampling_ratio = -1
        self.aligned = True

    def forward(self, data, boxes, boxes_num):
        return paddle.vision.ops.roi_align(
            data,
            boxes,
            boxes_num,
            output_size=self.output_size,
            spatial_scale=self.spatial_scale,
            sampling_ratio=self.sampling_ratio,
            aligned=self.aligned,
        )


@_test_only_pir
def test_roi_align():
    data = paddle.rand([1, 256, 32, 32])
    boxes = paddle.rand([3, 4])
    boxes[:, 2] += boxes[:, 0] + 3
    boxes[:, 3] += boxes[:, 1] + 4
    boxes_num = paddle.to_tensor([3]).astype("int32")
    op = BaseNet()
    op.eval()
    obj = APIOnnx(op, "roi_align", [17])
    obj.set_input_data("input_data", (data, boxes, boxes_num))
    obj.run()


if __name__ == "__main__":
    test_roi_align()
