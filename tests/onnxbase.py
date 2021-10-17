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

import os
import numpy as np
import logging
import paddle
from onnxruntime import InferenceSession


def compare(result, expect, delta=1e-10, rtol=1e-10):
    """
    比较函数
    :param result: 输入值
    :param expect: 输出值
    :param delta: 误差值
    :return:
    """
    if type(result) == np.ndarray:
        expect = np.array(expect)
        res = np.allclose(result, expect, atol=delta, rtol=rtol, equal_nan=True)
        # 出错打印错误数据
        if res is False:
            logging.error("the result is {}".format(result))
            logging.error("the expect is {}".format(expect))
        assert res
        assert result.shape == expect.shape
        assert result.dtype == expect.dtype
    elif type(result) == list:
        for i in range(len(result)):
            if isinstance(result[i], (np.generic, np.ndarray)):
                compare(result[i], expect[i], delta, rtol)
            else:
                compare(result[i].numpy(), expect[i], delta, rtol)


def randtool(dtype, low, high, shape):
    """
    np random tools
    """
    if dtype == "int":
        return np.random.randint(low, high, shape)

    elif dtype == "float":
        return low + (high - low) * np.random.random(shape)


class APIOnnx(object):
    """
     paddle API transfer to onnx
    """

    def __init__(self, func, name, ver_list, delta=1e-6, rtol=1e-5):
        self.seed = 33
        np.random.seed(self.seed)
        paddle.seed(self.seed)
        self.func = func
        if paddle.device.is_compiled_with_cuda() is True:
            self.places = ['gpu', 'cpu']
        else:
            self.places = ['cpu']
        self.name = name
        self._version = ver_list
        self.pwd = os.getcwd()
        self.delta = delta
        self.rtol = rtol
        self.static = False
        self.kwargs_dict = {"input_data": ()}
        self._shape = []
        self._dtype = []
        self.input_spec = []
        self.input_feed = {}

    def set_input_data(self, group_name, *args):
        """
        params dict tool
        """
        self.kwargs_dict[group_name] = args
        i = 0
        for in_data in self.kwargs_dict[group_name]:
            self.input_spec.append(
                paddle.static.InputSpec(
                    shape=in_data.shape, dtype=in_data.dtype, name=str(i)))
            self.input_feed[str(i)] = in_data.numpy()
            i += 1

    def _mkdir(self):
        """
        make dir to save all
        """
        save_path = os.path.join(self.pwd, self.name)
        if not os.path.exists(save_path):
            os.mkdir(save_path)

    def _mk_dygraph_exp(self):
        """
        make expect npy
        """
        return self.func(*self.kwargs_dict["input_data"])

    def _dygraph_to_onnx(self, ver):
        """
        paddle dygraph layer to onnx
        """
        paddle.onnx.export(
            self.func,
            os.path.join(self.pwd, self.name, self.name + str(ver)),
            input_spec=self.input_spec,
            opset_version=ver,
            enable_onnx_checker=True)

    def _dygraph_jit_save(self):
        """
        paddle dygraph layer to paddle static
        """
        paddle.jit.save(
            self.func,
            os.path.join(self.pwd, self.name, self.name + '_jit_save'),
            input_spec=self.input_spec)

    def _mk_onnx_res(self, ver):
        """
        make onnx res
        """
        sess = InferenceSession(
            os.path.join(self.pwd, self.name, self.name + str(ver) + '.onnx'))
        ort_outs = sess.run(output_names=None, input_feed=self.input_feed)
        return ort_outs[0]

    def run(self):
        """
        1. use dygraph layer to make exp
        2. dygraph layer to onnx
        3. use onnx to make res
        4. compare diff
        """
        self._mkdir()
        for place in self.places:
            paddle.set_device(place)
            logging.info("begin to test device: {}".format(place))
            exp = self._mk_dygraph_exp()
            res_fict = {}
            # export onnx models and make onnx res
            for v in self._version:
                logging.info("export op version {} to onnx...".format(str(v)))
                self._dygraph_to_onnx(ver=v)
                logging.info("make op version {} res of onnx...".format(str(v)))
                res_fict[str(v)] = self._mk_onnx_res(ver=v)
            # compare dygraph exp with onnx res
            for v in self._version:
                logging.info("compare dygraph exp with onnx version {} res...".
                             format(str(v)))
                compare(res_fict[str(v)], exp, delta=self.delta, rtol=self.rtol)
                logging.info(
                    "comparing dygraph exp with onnx version {} res is done.".
                    format(str(v)))
            # dygraph model jit save
            if self.static is True and place == 'gpu':
                logging.info("start to jit save...")
                self._dygraph_jit_save()
                logging.info("jit save is already...")
