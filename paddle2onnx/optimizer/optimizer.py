#   Copyright (c) 2019  PaddlePaddle Authors. All Rights Reserved.
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

# TODO useless node remove

from __future__ import absolute_import
import onnx
from onnx.optimizer import optimize


class GraphOptimizer(object):
    optimizers_list = [
        'eliminate_deadend',
        'eliminate_nop_dropout',
        'eliminate_nop_monotone_argmax',
        'eliminate_nop_pad',
        'extract_constant_to_initializer',
        'eliminate_unused_initializer',
        'eliminate_nop_transpose',
        # disable this optimizer until https://github.com/onnx/optimizer/issues/3 gets fixed
        # 'fuse_add_bias_into_conv',
        'fuse_consecutive_concats',
        'fuse_consecutive_log_softmax',
        'fuse_consecutive_reduce_unsqueeze',
        'fuse_consecutive_squeezes',
        'fuse_consecutive_transposes',
        'fuse_matmul_add_bias_into_gemm',
        'fuse_pad_into_conv',
        'fuse_transpose_into_gemm'
    ]

    def __init__(self):
        super(GraphOptimizer, self).__init__()
        self.skip_fuse_bn = True
        self.skipped_optimizers = []

    def optimize(self, onnx_model):
        if onnx_model.graph.node[-1].op_type != 'Identity':
            self.optimizers_list.append('eliminate_identity')
        if not self.skip_fuse_bn:
            self.optimizers_list.append('fuse_bn_into_conv')
        if self.skipped_optimizers is not None:
            for opt in self.skipped_optimizers:
                try:
                    self.optimizers_list.remove(opt)
                except ValueError:
                    pass

        onnx_model = optimize(
            onnx_model, self.optimizers_list, fixed_point=True)
        return onnx_model
