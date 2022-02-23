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

import logging

import os
import sys

import numpy as np
import yaml
import argparse
import psutil
import paddle2onnx
import onnx
from onnxruntime import InferenceSession
import paddle


def compare(result, expect, delta=1e-10, rtol=1e-10):
    """
    比较函数
    :param result: 输入值
    :param expect: 输出值
    :param delta: 误差值
    :return:
    """
    if type(result) == np.ndarray:
        if type(expect) == list:
            expect = expect[0]
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
        # assert res
        # assert result.shape == expect.shape
        # assert result.dtype == expect.dtype
        failed_type = []
        if not res:
            failed_type.append(" results has diff ")
        if not result.shape == expect.shape:
            failed_type.append(" shape is not equal ")
        if not result.dtype == expect.dtype:
            failed_type.append(" dtype is not equal ")
        return failed_type
    elif type(result) == list or type(result) == tuple:
        for i in range(len(result)):
            if isinstance(result[i], (np.generic, np.ndarray)):
                return compare(result[i], expect[i], delta, rtol)
            else:
                return compare(result[i].numpy(), expect[i], delta, rtol)


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


def str2bool(v):
    if v.lower() == 'true':
        return True
    else:
        return False


def str2list(v):
    if len(v) == 0:
        return []

    return [list(map(int, item.split(","))) for item in v.split(":")]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_shape', type=str2list, default=[])
    parser.add_argument('--input_dtype', type=str2list, default=["float32"])
    parser.add_argument('--model_dir', type=str, default="./")
    parser.add_argument('--save_file', type=str, default="model.onnx")
    parser.add_argument(
        '--paddle_model_file', type=str, default="model.pdmodel")
    parser.add_argument(
        '--paddle_params_file', type=str, default="model.pdiparams")
    parser.add_argument('--opset_version', type=int, default=15)
    args = parser.parse_args()
    return args


class ConverterAndChecher():
    def __init__(self):
        self.config = None
        self.model_dir = "./"
        self.paddle_model_file = "model.pdmodel"
        self.paddle_params_file = "model.pdiparam"
        self.opset_version = 15
        self.input_shape = [1, 3, 224, 224]
        self.input_dtype = ["floa32"]

    def load(self, config):
        self.config = config
        self.model_dir = config.model_dir
        self.paddle_model_file = config.paddle_model_file
        self.paddle_params_file = config.paddle_params_file
        self.save_file = config.save_file
        self.opset_version = config.opset_version
        self.input_shape = config.input_shape
        self.input_dtype = config.input_dtype
        self.model_filename = self.model_dir + "/" + self.paddle_model_file
        self.params_filename = self.model_dir + "/" + self.paddle_params_file
        self.onnx_model_file = self.model_dir + "/" + self.save_file

    def convert(self):
        paddle2onnx.export(self.model_filename, self.params_filename,
                           self.onnx_model_file, self.opset_version)

    def onnx_pred(self, numpy_input):
        onnx_model = onnx.load(self.onnx_model_file)
        sess = InferenceSession(onnx_model.SerializeToString())
        onnx_input_dict = {}
        for i in range(len(sess.get_inputs())):
            name = sess.get_inputs()[i].name
            onnx_input_dict[name] = numpy_input[i]

        ort_outs = sess.run(output_names=None, input_feed=onnx_input_dict)
        if len(ort_outs) > 1:
            return ort_outs
        return ort_outs[0]

    def paddle_pred(self, numpy_input):
        file_name = self.paddle_model_file.split('.')[0]
        model = paddle.jit.load(self.model_dir + "/" + file_name)
        input_list = []
        for i in range(len(numpy_input)):
            one_input = paddle.to_tensor(numpy_input[i])
            input_list.append(one_input)
        result = model(*input_list)
        return result

    def check(self):
        numpy_input = []
        for i in range(len(self.input_shape)):
            shape = self.input_shape[i]
            if self.input_dtype[i].count('int'):
                one_input = randtool("int", -20, 20,
                                     shape).astype(self.input_dtype[i])
            if self.input_dtype[i].count('float'):
                one_input = randtool("float", -2, 2,
                                     shape).astype(self.input_dtype[i])
            numpy_input.append(one_input)

        onnx_pred = self.onnx_pred(numpy_input)
        paddle_pred = self.paddle_pred(numpy_input)
        failed_type = compare(onnx_pred, paddle_pred, delta=1e-5, rtol=1e-5)
        with open("result.txt", 'a+') as f:
            if not len(failed_type):
                f.write(self.model_dir + ": convert success! \n")
                print(">>>> ", self.model_dir, " has no diff ! ")
            for i in range(len(failed_type)):
                f.write(self.model_dir + ": " + failed_type[i] + "\n")
                print(">>>> ", self.model_dir, " failed_type: ", failed_type)


def main():
    args = parse_args()
    runner = ConverterAndChecher()
    runner.load(args)
    runner.convert()
    runner.check()


if __name__ == "__main__":
    main()
