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


class BaseBackend():
    def __init__(self):
        self.inputs = None
        self.outputs = None
        self.config = None

    def set_runner(self, config):
        raise NotImplementdError("BaseBackend:set_runner")

    def set_input(self, inputs):
        raise NotImplementdError("BaseBackend:set_input")

    def preprocess(self):
        raise NotImplementdError("BaseBackend:preprocess")

    def postprocess(self):
        raise NotImplementdError("BaseBackend:postprocess")

    def predict(self):
        raise NotImplementdError("BaseBackend:predict")
