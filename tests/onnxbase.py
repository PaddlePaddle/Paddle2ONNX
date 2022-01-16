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
from inspect import isfunction
import numpy as np
import logging
import paddle
from onnxruntime import InferenceSession
from paddle2onnx.convert import dygraph2onnx


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
            diff = abs(result - expect)
            logging.error("Output has diff! max diff: {}".format(np.amax(diff)))
        if result.dtype != expect.dtype:
            logging.error(
                "Different output data types! res type is: {}, and expect type is: {}".
                format(result.dtype, expect.dtype))
        assert res
        assert result.shape == expect.shape
        assert result.dtype == expect.dtype
    elif type(result) == list or type(result) == tuple:
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

    elif dtype == "bool":
        return np.random.randint(low, high, shape).astype("bool")


class BuildFunc(paddle.nn.Layer):
    """
    simple Net
    """

    def __init__(self, inner_func, **super_param):
        super(BuildFunc, self).__init__()
        self.inner_func = inner_func
        self._super_param = super_param

    def forward(self, inputs):
        """
        forward
        """
        x = self.inner_func(inputs, **self._super_param)
        return x


class BuildClass(paddle.nn.Layer):
    """
    simple Net
    """

    def __init__(self, inner_class, **super_param):
        super(BuildClass, self).__init__()
        self.inner_class = inner_class(**super_param)

    def forward(self, inputs):
        """
        forward
        """
        x = self.inner_class(inputs)
        return x


class APIOnnx(object):
    """
     paddle API transfer to onnx
    """

    def __init__(self,
                 func,
                 file_name,
                 ver_list,
                 ops=[],
                 input_spec_shape=[],
                 delta=1e-5,
                 rtol=1e-5,
                 **sup_params):
        self.ops = ops
        if isinstance(self.ops, str):
            self.ops = [self.ops]
        self.seed = 33
        np.random.seed(self.seed)
        paddle.seed(self.seed)
        self.func = func
        if paddle.device.is_compiled_with_cuda() is True:
            self.places = ['gpu', 'cpu']
        else:
            self.places = ['cpu']
        self.name = file_name
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
        self.input_spec_shape = input_spec_shape
        self.input_dtype = []

        if isfunction(self.func):
            # self._func = self.BuildFunc(self.func, **self.kwargs_dict_dygraph["params_group1"])
            self._func = BuildFunc(inner_func=self.func, **sup_params)
        elif isinstance(self.func, type):
            self._func = BuildClass(inner_class=self.func, **sup_params)
        else:
            self._func = self.func

    def set_input_data(self, group_name, *args):
        """
        params dict tool
        """
        self.kwargs_dict[group_name] = args
        if isinstance(self.kwargs_dict[group_name][0], tuple):
            self.kwargs_dict[group_name] = self.kwargs_dict[group_name][0]

        i = 0
        for in_data in self.kwargs_dict[group_name]:
            if isinstance(in_data, list):
                for tensor_data in in_data:
                    self.input_dtype.append(tensor_data.dtype)
                    self.input_spec.append(
                        paddle.static.InputSpec(
                            shape=tensor_data.shape,
                            dtype=tensor_data.dtype,
                            name=str(i)))
                    self.input_feed[str(i)] = tensor_data.numpy()
                    i += 1
            else:
                if isinstance(in_data, tuple):
                    in_data = in_data[0]
                self.input_dtype.append(in_data.dtype)
                self.input_spec.append(
                    paddle.static.InputSpec(
                        shape=in_data.shape, dtype=in_data.dtype, name=str(i)))
                self.input_feed[str(i)] = in_data.numpy()
                i += 1

    def set_input_spec(self):
        if len(self.input_spec_shape) == 0:
            return
        self.input_spec.clear()
        i = 0
        for shape in self.input_spec_shape:
            self.input_spec.append(
                paddle.static.InputSpec(
                    shape=shape, dtype=self.input_dtype[i], name=str(i)))
            i += 1

    def _mkdir(self):
        """
        make dir to save all
        """
        save_path = os.path.join(self.pwd, self.name)
        if not os.path.exists(save_path):
            os.mkdir(save_path)

    def _mk_dygraph_exp(self, instance):
        """
        make expect npy
        """
        return instance(*self.kwargs_dict["input_data"])

    def _dygraph_to_onnx(self, instance, ver):
        """
        paddle dygraph layer to onnx
        """
        paddle.onnx.export(
            instance,
            os.path.join(self.pwd, self.name, self.name + '_' + str(ver)),
            input_spec=self.input_spec,
            opset_version=ver,
            enable_onnx_checker=True)

    def _dygraph_jit_save(self, instance):
        """
        paddle dygraph layer to paddle static
        """
        paddle.jit.save(
            instance,
            os.path.join(self.pwd, self.name, self.name + '_jit_save'),
            input_spec=self.input_spec)

    def _mk_onnx_res(self, ver):
        """
        make onnx res
        """
        sess = InferenceSession(
            os.path.join(self.pwd, self.name, self.name + '_' + str(ver) +
                         '.onnx'))
        ort_outs = sess.run(output_names=None, input_feed=self.input_feed)
        if len(ort_outs) > 1:
            return ort_outs
        return ort_outs[0]

    def add_kwargs_to_dict(self, group_name, **kwargs):
        """
        params dict tool
        """
        self.kwargs_dict[group_name] = kwargs

    def check_ops(self, version):
        if len(self.ops) == 0:
            return
        paddle_graph = dygraph2onnx(
            self._func,
            "path",
            input_spec=self.input_spec,
            opset_version=version,
            get_paddle_graph=True)

        status = False
        for op in self.ops:
            for key, val in paddle_graph.node_map.items():
                if op in key:
                    status = True
        assert status is True, "{} op in not in convert OPs, all OPs :{}".format(
            self.ops, paddle_graph.node_map.keys())

    def run(self):
        """
        1. use dygraph layer to make exp
        2. dygraph layer to onnx
        3. use onnx to make res
        4. compare diff
        """
        self._mkdir()
        self.set_input_spec()
        for place in self.places:
            paddle.set_device(place)

            exp = self._mk_dygraph_exp(self._func)
            res_fict = {}
            # export onnx models and make onnx res
            for v in self._version:
                self.check_ops(v)
                self._dygraph_to_onnx(instance=self._func, ver=v)
                res_fict[str(v)] = self._mk_onnx_res(ver=v)

            for v in self._version:
                compare(res_fict[str(v)], exp, delta=self.delta, rtol=self.rtol)

            # dygraph model jit save
            if self.static is True and place == 'gpu':
                self._dygraph_jit_save(instance=self._func)
