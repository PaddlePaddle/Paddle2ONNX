# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

import paddle.fluid as fluid
import sys
import paddle

paddle.enable_static()

model_dir = sys.argv[1]
new_model_dir = sys.argv[2]
exe = fluid.Executor(fluid.CPUPlace())
[inference_program, feed_target_names, fetch_targets] = fluid.io.load_inference_model(
    dirname=model_dir, executor=exe
)

print(feed_target_names)
fluid.io.save_inference_model(
    dirname=new_model_dir,
    feeded_var_names=feed_target_names,
    target_vars=fetch_targets,
    executor=exe,
    main_program=inference_program,
    params_filename="__params__",
)
