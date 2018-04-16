# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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


def get_op_io_info(op):
    inputs = dict([(name, op.input(name)) for name in op.input_names])
    attrs = dict(
        [(name, op.attr(name))
         for name in op.attr_names]) if op.attr_names is not None else None
    outputs = dict([(name, op.output(name)) for name in op.output_names])

    return inputs, attrs, outputs
