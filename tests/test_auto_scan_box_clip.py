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
np.random.seed(33)


def test_box_clip_0():
    main_program = paddle.static.Program()
    startup_program = paddle.static.Program()
    with paddle.static.program_guard(main_program, startup_program):
        boxes = fluid.data(
            name='boxes', shape=[-1, 8, 4], dtype='float32', lod_level=1)
        im_info = fluid.data(name='im_info', shape=[-1, 3])
        out = fluid.layers.box_clip(input=boxes, im_info=im_info)

        exe = paddle.static.Executor(paddle.CPUPlace())
        exe.run(paddle.static.default_startup_program())
        input_data = randtool("int", 1, 100, [1, 8, 4]).astype('float32')
        im_info_data = randtool("float", 1, 10, [1, 3]).astype('float32')
        im_info_data[:, 2] = randtool("float", 0.1, 3, [1]).astype('float32')
        t1 = fluid.create_lod_tensor(input_data, [[1]], fluid.CPUPlace())
        t2 = fluid.create_lod_tensor(im_info_data, [[1]], fluid.CPUPlace())
        result, = exe.run(feed={"boxes": t1,
                                "im_info": t2},
                          fetch_list=[out],
                          return_numpy=False)
        result = np.array(result)
        path_prefix = "./box_clip"
        fluid.io.save_inference_model(path_prefix, ["boxes", "im_info"], [out],
                                      exe)
        onnx_path = path_prefix + "./model.onnx"
        program2onnx(
            model_dir=path_prefix,
            save_file=onnx_path,
            opset_version=11,
            enable_onnx_checker=True)
        sess = rt.InferenceSession(onnx_path)
        input_name1 = sess.get_inputs()[0].name
        input_name2 = sess.get_inputs()[1].name
        label_name = sess.get_outputs()[0].name
        pred_onnx = sess.run(
            None, {input_name1: input_data,
                   input_name2: im_info_data})[0]
        pred_onnx = np.array(pred_onnx)
        compare(pred_onnx, result, 1e-5, 1e-5)


def test_box_clip_1():
    main_program = paddle.static.Program()
    startup_program = paddle.static.Program()
    with paddle.static.program_guard(main_program, startup_program):
        boxes = fluid.data(
            name='boxes', shape=[-1, 15, 8, 4], dtype='float32', lod_level=1)
        im_info = fluid.data(name='im_info', shape=[-1, 3])
        out = fluid.layers.box_clip(input=boxes, im_info=im_info)

        exe = paddle.static.Executor(paddle.CPUPlace())
        exe.run(paddle.static.default_startup_program())
        input_data = randtool("int", 1, 100, [10, 15, 8, 4]).astype('float32')
        im_info_data = randtool("float", 1, 10, [1, 3]).astype('float32')
        im_info_data[:, 2] = randtool("float", 0.1, 3, [1]).astype('float32')
        t1 = fluid.create_lod_tensor(input_data, [[10]], fluid.CPUPlace())
        t2 = fluid.create_lod_tensor(im_info_data, [[1]], fluid.CPUPlace())
        result, = exe.run(feed={"boxes": t1,
                                "im_info": t2},
                          fetch_list=[out],
                          return_numpy=False)
        result = np.array(result)
        path_prefix = "./box_clip"
        fluid.io.save_inference_model(path_prefix, ["boxes", "im_info"], [out],
                                      exe)
        onnx_path = path_prefix + "./model.onnx"
        program2onnx(
            model_dir=path_prefix,
            save_file=onnx_path,
            opset_version=11,
            enable_onnx_checker=True)
        sess = rt.InferenceSession(onnx_path)
        input_name1 = sess.get_inputs()[0].name
        input_name2 = sess.get_inputs()[1].name
        label_name = sess.get_outputs()[0].name
        pred_onnx = sess.run(
            None, {input_name1: input_data,
                   input_name2: im_info_data})[0]
        pred_onnx = np.array(pred_onnx)
        compare(pred_onnx, result, 1e-5, 1e-5)


def test_box_clip_2():
    main_program = paddle.static.Program()
    startup_program = paddle.static.Program()
    with paddle.static.program_guard(main_program, startup_program):
        boxes = fluid.data(
            name='boxes', shape=[-1, 15, 8, 4], dtype='float32', lod_level=1)
        im_info = fluid.data(name='im_info', shape=[-1, 3])
        out = fluid.layers.box_clip(input=boxes, im_info=im_info)

        exe = paddle.static.Executor(paddle.CPUPlace())
        exe.run(paddle.static.default_startup_program())
        input_data = randtool("int", 1, 100, [10, 15, 8, 4]).astype('float32')
        im_info_data = randtool("float", 1, 10, [1, 3]).astype('float32')
        im_info_data[:, 2] = randtool("float", 0.1, 3, [1]).astype('float32')
        t1 = fluid.create_lod_tensor(input_data, [[10]], fluid.CPUPlace())
        t2 = fluid.create_lod_tensor(im_info_data, [[1]], fluid.CPUPlace())
        result, = exe.run(feed={"boxes": t1,
                                "im_info": t2},
                          fetch_list=[out],
                          return_numpy=False)
        result = np.array(result)
        path_prefix = "./box_clip"
        fluid.io.save_inference_model(path_prefix, ["boxes", "im_info"], [out],
                                      exe)
        onnx_path = path_prefix + "./model.onnx"
        program2onnx(
            model_dir=path_prefix,
            save_file=onnx_path,
            opset_version=13,
            enable_onnx_checker=True)
        sess = rt.InferenceSession(onnx_path)
        input_name1 = sess.get_inputs()[0].name
        input_name2 = sess.get_inputs()[1].name
        label_name = sess.get_outputs()[0].name
        pred_onnx = sess.run(
            None, {input_name1: input_data,
                   input_name2: im_info_data})[0]
        pred_onnx = np.array(pred_onnx)
        compare(pred_onnx, result, 1e-5, 1e-5)


def test_box_clip_3():
    main_program = paddle.static.Program()
    startup_program = paddle.static.Program()
    with paddle.static.program_guard(main_program, startup_program):
        boxes = fluid.data(
            name='boxes',
            shape=[-1, 30, 15, 8, 4],
            dtype='float32',
            lod_level=1)
        im_info = fluid.data(name='im_info', shape=[-1, 3])
        out = fluid.layers.box_clip(input=boxes, im_info=im_info)

        exe = paddle.static.Executor(paddle.CPUPlace())
        exe.run(paddle.static.default_startup_program())
        input_data = randtool("int", 1, 10,
                              [10, 30, 15, 8, 4]).astype('float32')
        im_info_data = randtool("float", 1, 4, [1, 3]).astype('float32')
        im_info_data[:, 2] = randtool("float", 0.1, 3, [1]).astype('float32')
        t1 = fluid.create_lod_tensor(input_data, [[10]], fluid.CPUPlace())
        t2 = fluid.create_lod_tensor(im_info_data, [[1]], fluid.CPUPlace())
        result, = exe.run(feed={"boxes": t1,
                                "im_info": t2},
                          fetch_list=[out],
                          return_numpy=False)
        result = np.array(result)
        path_prefix = "./box_clip"
        fluid.io.save_inference_model(path_prefix, ["boxes", "im_info"], [out],
                                      exe)
        onnx_path = path_prefix + "./model.onnx"
        program2onnx(
            model_dir=path_prefix,
            save_file=onnx_path,
            opset_version=15,
            enable_onnx_checker=True)
        sess = rt.InferenceSession(onnx_path)
        input_name1 = sess.get_inputs()[0].name
        input_name2 = sess.get_inputs()[1].name
        label_name = sess.get_outputs()[0].name
        pred_onnx = sess.run(
            None, {input_name1: input_data,
                   input_name2: im_info_data})[0]
        pred_onnx = np.array(pred_onnx)
        compare(pred_onnx, result, 1e-5, 1e-5)
