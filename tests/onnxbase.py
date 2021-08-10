#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
tools
"""
import os
import numpy as np
import logging
import paddle
from onnxruntime import InferenceSession


def compare(result, expect, delta=1e-6, rtol=1e-5):
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

    def __init__(self,
                 func,
                 name,
                 input_shape,
                 ver_list,
                 delta=1e-6,
                 rtol=1e-5,
                 binputs=False,
                 data_type='float32'):
        self.seed = 33
        np.random.seed(self.seed)
        self.delta = 1e-6
        self.rtol = 1e-5
        self.func = func
        if paddle.device.is_compiled_with_cuda() is True:
            self.places = ['gpu', 'cpu']
        else:
            self.places = ['cpu']
        self.name = name
        self._version = ver_list
        self.pwd = os.getcwd()
        self.input_spec = paddle.static.InputSpec(
            shape=input_shape, dtype=data_type, name='image')
        self.input_spec_ = paddle.static.InputSpec(
            shape=input_shape, dtype=data_type, name='image_')
        self.input_data = randtool("float", -1, 1,
                                   input_shape).astype(data_type)
        self.input_data_ = randtool("float", 0, 2,
                                    input_shape).astype(data_type)
        self.delta = delta
        self.rtol = rtol
        self.binputs = binputs
        self.static = True

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
        in_tensor = paddle.to_tensor(self.input_data)
        if self.binputs is False:
            return self.func(in_tensor).numpy()
        else:
            in_tensor_ = paddle.to_tensor(self.input_data_)
            return self.func(in_tensor, in_tensor_).numpy()

    def _dygraph_to_onnx(self, ver):
        """
        paddle dygraph layer to onnx
        """
        if self.binputs is False:
            paddle.onnx.export(
                self.func,
                os.path.join(self.pwd, self.name, self.name + str(ver)),
                input_spec=[self.input_spec],
                opset_version=ver,
                enable_onnx_checker=True)
        else:
            paddle.onnx.export(
                self.func,
                os.path.join(self.pwd, self.name, self.name + str(ver)),
                input_spec=[self.input_spec, self.input_spec_],
                opset_version=ver,
                enable_onnx_checker=True)

    def _dygraph_jit_save(self):
        """
        paddle dygraph layer to paddle static
        """
        if self.binputs is False:
            paddle.jit.save(
                self.func,
                os.path.join(self.pwd, self.name, self.name + '_jit_save'),
                input_spec=[self.input_spec])
        else:
            paddle.jit.save(
                self.func,
                os.path.join(self.pwd, self.name, self.name + '_jit_save'),
                input_spec=[self.input_spec, self.input_spec])

    def _mk_onnx_res(self, ver):
        """
        make onnx res
        """
        sess = InferenceSession(
            os.path.join(self.pwd, self.name, self.name + str(ver) + '.onnx'))
        if self.binputs is False:
            ort_outs = sess.run(output_names=None,
                                input_feed={'image': self.input_data})
        else:
            ort_outs = sess.run(output_names=None,
                                input_feed={
                                    'image': self.input_data,
                                    'image_': self.input_data_
                                })
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

            exp = self._mk_dygraph_exp()
            res_fict = {}
            # export onnx models and make onnx res
            for v in self._version:
                logging.info("export op version {} to onnx...".format(str(v)))
                self._dygraph_to_onnx(ver=v)
                logging.info("make op version {} res of onnx...".format(str(v)))
                res_fict[str(v)] = self._mk_onnx_res(ver=v)
                # print(res_fict[str(v)])
                # print(exp)
                # print('******' * 30)
            # compare dygraph exp with onnx res
            for v in self._version:
                logging.info("compare dygraph exp with onnx version {} res...".
                             format(str(v)))
                compare(res_fict[str(v)], exp, delta=self.delta, rtol=self.rtol)

            # dygraph model jit save
            if self.static is True:
                self._dygraph_jit_save()
