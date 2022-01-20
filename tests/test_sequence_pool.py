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

from auto_scan_test import OPConvertAutoScanTest, BaseNet
from hypothesis import reproduce_failure
import hypothesis.strategies as st
import numpy as np
import unittest
import paddle
import paddle.fluid as fluid
from paddle2onnx.command import program2onnx
import logging
from onnxbase import randtool, compare
import onnxruntime as rt
paddle.enable_static()


def test_sequence_pool_sum_0():
    """
    api: paddle.static.nn.sequence_pool
    op version: 11
    """
    main_program = paddle.static.Program()
    startup_program = paddle.static.Program()
    with paddle.static.program_guard(main_program, startup_program):
        x = paddle.static.data(
            name='x', shape=[5], dtype='float32', lod_level=1)
        sum_x = paddle.static.nn.sequence_pool(
            x, pool_type="sum", pad_value=1, is_test=True)
        exe = paddle.static.Executor(paddle.CPUPlace())
        exe.run(paddle.static.default_startup_program())
        data = randtool("float", -1, 1, [5]).astype('float32')
        t = fluid.create_lod_tensor(data, [[5]], fluid.CPUPlace())
        result, = exe.run(feed={"x": t}, fetch_list=[sum_x])
        path_prefix = "./pool_sum"
        fluid.io.save_inference_model(path_prefix, ["x"], [sum_x], exe)
        onnx_path = path_prefix + "./model.onnx"
        program2onnx(
            model_dir=path_prefix,
            save_file=onnx_path,
            opset_version=11,
            enable_onnx_checker=True)
        sess = rt.InferenceSession(onnx_path)
        input_name = sess.get_inputs()[0].name
        label_name = sess.get_outputs()[0].name
        pred_onnx = sess.run(None, {input_name: data})[0]
        compare(pred_onnx, result, 1e-5, 1e-5)


def test_sequence_pool_sum_1():
    """
    api: paddle.static.nn.sequence_pool
    op version: 11
    """
    main_program = paddle.static.Program()
    startup_program = paddle.static.Program()
    with paddle.static.program_guard(main_program, startup_program):
        x = paddle.static.data(
            name='x', shape=[5, 20], dtype='float32', lod_level=1)
        sum_x = paddle.static.nn.sequence_pool(
            x, pool_type="sum", pad_value=1, is_test=True)
        exe = paddle.static.Executor(paddle.CPUPlace())
        exe.run(paddle.static.default_startup_program())
        data = randtool("float", -1, 1, [5, 20]).astype('float32')
        t = fluid.create_lod_tensor(data, [[5]], fluid.CPUPlace())
        result, = exe.run(feed={"x": t}, fetch_list=[sum_x])
        path_prefix = "./pool_sum"
        fluid.io.save_inference_model(path_prefix, ["x"], [sum_x], exe)
        onnx_path = path_prefix + "./model.onnx"
        program2onnx(
            model_dir=path_prefix,
            save_file=onnx_path,
            opset_version=11,
            enable_onnx_checker=True)
        sess = rt.InferenceSession(onnx_path)
        input_name = sess.get_inputs()[0].name
        label_name = sess.get_outputs()[0].name
        pred_onnx = sess.run(None, {input_name: data})[0]
        compare(pred_onnx, result, 1e-5, 1e-5)


def test_sequence_pool_sum_2():
    """
    api: paddle.static.nn.sequence_pool
    op version: 11
    """
    main_program = paddle.static.Program()
    startup_program = paddle.static.Program()
    with paddle.static.program_guard(main_program, startup_program):
        x = paddle.static.data(
            name='x', shape=[5, 10, 5], dtype='float32', lod_level=1)
        sum_x = paddle.static.nn.sequence_pool(
            x, pool_type="sum", pad_value=1, is_test=True)
        exe = paddle.static.Executor(paddle.CPUPlace())
        exe.run(paddle.static.default_startup_program())
        data = randtool("float", -1, 1, [5, 10, 5]).astype('float32')
        t = fluid.create_lod_tensor(data, [[5]], fluid.CPUPlace())
        result, = exe.run(feed={"x": t}, fetch_list=[sum_x])
        path_prefix = "./pool_sum"
        fluid.io.save_inference_model(path_prefix, ["x"], [sum_x], exe)
        onnx_path = path_prefix + "./model.onnx"
        program2onnx(
            model_dir=path_prefix,
            save_file=onnx_path,
            opset_version=11,
            enable_onnx_checker=True)
        sess = rt.InferenceSession(onnx_path)
        input_name = sess.get_inputs()[0].name
        label_name = sess.get_outputs()[0].name
        pred_onnx = sess.run(None, {input_name: data})[0]
        compare(pred_onnx, result, 1e-5, 1e-5)


def test_sequence_pool_sum_3():
    """
    api: paddle.static.nn.sequence_pool
    op version: 11
    """
    main_program = paddle.static.Program()
    startup_program = paddle.static.Program()
    with paddle.static.program_guard(main_program, startup_program):
        x = paddle.static.data(
            name='x', shape=[5, 10, 5, 20], dtype='float32', lod_level=1)
        sum_x = paddle.static.nn.sequence_pool(
            x, pool_type="sum", pad_value=1, is_test=True)
        exe = paddle.static.Executor(paddle.CPUPlace())
        exe.run(paddle.static.default_startup_program())
        data = randtool("float", -1, 1, [5, 10, 5, 20]).astype('float32')
        t = fluid.create_lod_tensor(data, [[5]], fluid.CPUPlace())
        result, = exe.run(feed={"x": t}, fetch_list=[sum_x])
        path_prefix = "./pool_sum"
        fluid.io.save_inference_model(path_prefix, ["x"], [sum_x], exe)
        onnx_path = path_prefix + "./model.onnx"
        program2onnx(
            model_dir=path_prefix,
            save_file=onnx_path,
            opset_version=11,
            enable_onnx_checker=True)
        sess = rt.InferenceSession(onnx_path)
        input_name = sess.get_inputs()[0].name
        label_name = sess.get_outputs()[0].name
        pred_onnx = sess.run(None, {input_name: data})[0]
        compare(pred_onnx, result, 1e-5, 1e-5)


def test_sequence_pool_average_1():
    """
    api: paddle.static.nn.sequence_pool
    op version: 11
    """
    main_program = paddle.static.Program()
    startup_program = paddle.static.Program()
    with paddle.static.program_guard(main_program, startup_program):
        x = paddle.static.data(
            name='x', shape=[5, 10, 5, 20], dtype='float32', lod_level=1)
        avg_x = paddle.static.nn.sequence_pool(
            x, pool_type="average", pad_value=1, is_test=True)
        exe = paddle.static.Executor(paddle.CPUPlace())
        exe.run(paddle.static.default_startup_program())
        data = randtool("float", -1, 1, [5, 10, 5, 20]).astype('float32')
        t = fluid.create_lod_tensor(data, [[5]], fluid.CPUPlace())
        result, = exe.run(feed={"x": t}, fetch_list=[avg_x])
        path_prefix = "./pool_sum"
        fluid.io.save_inference_model(path_prefix, ["x"], [avg_x], exe)
        onnx_path = path_prefix + "./model.onnx"
        program2onnx(
            model_dir=path_prefix,
            save_file=onnx_path,
            opset_version=11,
            enable_onnx_checker=True)
        sess = rt.InferenceSession(onnx_path)
        input_name = sess.get_inputs()[0].name
        label_name = sess.get_outputs()[0].name
        pred_onnx = sess.run(None, {input_name: data})[0]
        compare(pred_onnx, result, 1e-5, 1e-5)


def test_sequence_pool_sqrt_1():
    """
    api: paddle.static.nn.sequence_pool
    op version: 11
    """
    main_program = paddle.static.Program()
    startup_program = paddle.static.Program()
    with paddle.static.program_guard(main_program, startup_program):
        x = paddle.static.data(
            name='x', shape=[5, 10, 5, 20], dtype='float32', lod_level=1)
        sqrt_x = paddle.static.nn.sequence_pool(
            x, pool_type="sqrt", pad_value=1, is_test=True)
        exe = paddle.static.Executor(paddle.CPUPlace())
        exe.run(paddle.static.default_startup_program())
        data = randtool("float", -1, 1, [5, 10, 5, 20]).astype('float32')
        t = fluid.create_lod_tensor(data, [[5]], fluid.CPUPlace())
        result, = exe.run(feed={"x": t}, fetch_list=[sqrt_x])
        path_prefix = "./pool_sum"
        fluid.io.save_inference_model(path_prefix, ["x"], [sqrt_x], exe)
        onnx_path = path_prefix + "./model.onnx"
        program2onnx(
            model_dir=path_prefix,
            save_file=onnx_path,
            opset_version=11,
            enable_onnx_checker=True)
        sess = rt.InferenceSession(onnx_path)
        input_name = sess.get_inputs()[0].name
        label_name = sess.get_outputs()[0].name
        pred_onnx = sess.run(None, {input_name: data})[0]
        compare(pred_onnx, result, 1e-5, 1e-5)
