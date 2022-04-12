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


def test_grid_sample_align_corners():
    main_program = paddle.static.Program()
    startup_program = paddle.static.Program()
    # onnxruntime 中的Floor不支持float64
    dtype = 'float32'
    align_corners = True
    N = 5
    with paddle.static.program_guard(main_program, startup_program):
        x = fluid.data(
            name='x', shape=[-1, 6, -1, -1], dtype=dtype, lod_level=1)
        grid = fluid.data(
            name='grid', shape=[-1, 3, -1, -1], dtype=dtype, lod_level=1)
        out = paddle.nn.functional.grid_sample(
            x, grid, align_corners=align_corners)

        exe = paddle.static.Executor(paddle.CPUPlace())
        exe.run(paddle.static.default_startup_program())
        x_data = randtool("int", 1, 10, [N, 6, 3, 3]).astype(dtype)
        grid_data = randtool("float", 1, 10, [N, 3, 4, 2]).astype(dtype)
        x_val = fluid.create_lod_tensor(x_data, [[N]], fluid.CPUPlace())
        grid_val = fluid.create_lod_tensor(grid_data, [[N]], fluid.CPUPlace())
        result, = exe.run(feed={"x": x_val,
                                "grid": grid_val},
                          fetch_list=[out],
                          return_numpy=False)
        result = np.array(result)
        path_prefix = "./grid_sampler"
        fluid.io.save_inference_model(path_prefix, ["x", "grid"], [out], exe)
        onnx_path = path_prefix + "/model.onnx"
        program2onnx(
            model_dir=path_prefix,
            save_file=onnx_path,
            opset_version=11,
            enable_onnx_checker=True)

        sess = rt.InferenceSession(onnx_path)
        input_name1 = sess.get_inputs()[0].name
        input_name2 = sess.get_inputs()[1].name
        label_name = sess.get_outputs()[0].name
        pred_onnx = sess.run([label_name],
                             {input_name1: x_data,
                              input_name2: grid_data})[0]
        pred_onnx = np.array(pred_onnx)
        compare(pred_onnx, result, 1e-5, 1e-5)


def test_grid_sample_align_corners_False():
    main_program = paddle.static.Program()
    startup_program = paddle.static.Program()
    # onnxruntime 中的Floor不支持float64
    dtype = 'float32'
    align_corners = False
    N = 5
    with paddle.static.program_guard(main_program, startup_program):
        x = fluid.data(
            name='x', shape=[-1, 6, -1, -1], dtype=dtype, lod_level=1)
        grid = fluid.data(
            name='grid', shape=[-1, 3, -1, -1], dtype=dtype, lod_level=1)
        out = paddle.nn.functional.grid_sample(
            x, grid, align_corners=align_corners)

        exe = paddle.static.Executor(paddle.CPUPlace())
        exe.run(paddle.static.default_startup_program())
        x_data = randtool("int", 1, 10, [N, 6, 3, 3]).astype(dtype)
        grid_data = randtool("float", 1, 10, [N, 3, 4, 2]).astype(dtype)
        x_val = fluid.create_lod_tensor(x_data, [[N]], fluid.CPUPlace())
        grid_val = fluid.create_lod_tensor(grid_data, [[N]], fluid.CPUPlace())
        result, = exe.run(feed={"x": x_val,
                                "grid": grid_val},
                          fetch_list=[out],
                          return_numpy=False)
        result = np.array(result)
        path_prefix = "./grid_sampler"
        fluid.io.save_inference_model(path_prefix, ["x", "grid"], [out], exe)
        onnx_path = path_prefix + "/model.onnx"
        program2onnx(
            model_dir=path_prefix,
            save_file=onnx_path,
            opset_version=11,
            enable_onnx_checker=True)

        sess = rt.InferenceSession(onnx_path)
        input_name1 = sess.get_inputs()[0].name
        input_name2 = sess.get_inputs()[1].name
        label_name = sess.get_outputs()[0].name
        pred_onnx = sess.run([label_name],
                             {input_name1: x_data,
                              input_name2: grid_data})[0]
        pred_onnx = np.array(pred_onnx)
        compare(pred_onnx, result, 1e-5, 1e-5)


def test_grid_sample_align_corners_Opset12():
    main_program = paddle.static.Program()
    startup_program = paddle.static.Program()
    # onnxruntime 中的Floor不支持float64
    dtype = 'float32'
    align_corners = False
    N = 5
    with paddle.static.program_guard(main_program, startup_program):
        x = fluid.data(
            name='x', shape=[-1, 6, -1, -1], dtype=dtype, lod_level=1)
        grid = fluid.data(
            name='grid', shape=[-1, 3, -1, -1], dtype=dtype, lod_level=1)
        out = paddle.nn.functional.grid_sample(
            x, grid, align_corners=align_corners)

        exe = paddle.static.Executor(paddle.CPUPlace())
        exe.run(paddle.static.default_startup_program())
        x_data = randtool("int", 1, 10, [N, 6, 3, 3]).astype(dtype)
        grid_data = randtool("float", 1, 10, [N, 3, 4, 2]).astype(dtype)
        x_val = fluid.create_lod_tensor(x_data, [[N]], fluid.CPUPlace())
        grid_val = fluid.create_lod_tensor(grid_data, [[N]], fluid.CPUPlace())
        result, = exe.run(feed={"x": x_val,
                                "grid": grid_val},
                          fetch_list=[out],
                          return_numpy=False)
        result = np.array(result)
        path_prefix = "./grid_sampler"
        fluid.io.save_inference_model(path_prefix, ["x", "grid"], [out], exe)
        onnx_path = path_prefix + "/model.onnx"
        program2onnx(
            model_dir=path_prefix,
            save_file=onnx_path,
            opset_version=12,
            enable_onnx_checker=True)

        sess = rt.InferenceSession(onnx_path)
        input_name1 = sess.get_inputs()[0].name
        input_name2 = sess.get_inputs()[1].name
        label_name = sess.get_outputs()[0].name
        pred_onnx = sess.run([label_name],
                             {input_name1: x_data,
                              input_name2: grid_data})[0]
        pred_onnx = np.array(pred_onnx)
        compare(pred_onnx, result, 1e-5, 1e-5)


def test_grid_sample_align_corners_Opset13():
    main_program = paddle.static.Program()
    startup_program = paddle.static.Program()
    # onnxruntime 中的Floor不支持float64
    dtype = 'float32'
    align_corners = False
    N = 5
    with paddle.static.program_guard(main_program, startup_program):
        x = fluid.data(
            name='x', shape=[-1, 6, -1, -1], dtype=dtype, lod_level=1)
        grid = fluid.data(
            name='grid', shape=[-1, 3, -1, -1], dtype=dtype, lod_level=1)
        out = paddle.nn.functional.grid_sample(
            x, grid, align_corners=align_corners)

        exe = paddle.static.Executor(paddle.CPUPlace())
        exe.run(paddle.static.default_startup_program())
        x_data = randtool("int", 1, 10, [N, 6, 3, 3]).astype(dtype)
        grid_data = randtool("float", 1, 10, [N, 3, 4, 2]).astype(dtype)
        x_val = fluid.create_lod_tensor(x_data, [[N]], fluid.CPUPlace())
        grid_val = fluid.create_lod_tensor(grid_data, [[N]], fluid.CPUPlace())
        result, = exe.run(feed={"x": x_val,
                                "grid": grid_val},
                          fetch_list=[out],
                          return_numpy=False)
        result = np.array(result)
        path_prefix = "./grid_sampler"
        fluid.io.save_inference_model(path_prefix, ["x", "grid"], [out], exe)
        onnx_path = path_prefix + "/model.onnx"
        program2onnx(
            model_dir=path_prefix,
            save_file=onnx_path,
            opset_version=13,
            enable_onnx_checker=True)

        sess = rt.InferenceSession(onnx_path)
        input_name1 = sess.get_inputs()[0].name
        input_name2 = sess.get_inputs()[1].name
        label_name = sess.get_outputs()[0].name
        pred_onnx = sess.run([label_name],
                             {input_name1: x_data,
                              input_name2: grid_data})[0]
        pred_onnx = np.array(pred_onnx)
        compare(pred_onnx, result, 1e-5, 1e-5)


def test_grid_sample_align_corners_Opset14():
    main_program = paddle.static.Program()
    startup_program = paddle.static.Program()
    # onnxruntime 中的Floor不支持float64
    dtype = 'float32'
    align_corners = False
    N = 5
    with paddle.static.program_guard(main_program, startup_program):
        x = fluid.data(
            name='x', shape=[-1, 6, -1, -1], dtype=dtype, lod_level=1)
        grid = fluid.data(
            name='grid', shape=[-1, 3, -1, -1], dtype=dtype, lod_level=1)
        out = paddle.nn.functional.grid_sample(
            x, grid, align_corners=align_corners)

        exe = paddle.static.Executor(paddle.CPUPlace())
        exe.run(paddle.static.default_startup_program())
        x_data = randtool("int", 1, 10, [N, 6, 3, 3]).astype(dtype)
        grid_data = randtool("float", 1, 10, [N, 3, 4, 2]).astype(dtype)
        x_val = fluid.create_lod_tensor(x_data, [[N]], fluid.CPUPlace())
        grid_val = fluid.create_lod_tensor(grid_data, [[N]], fluid.CPUPlace())
        result, = exe.run(feed={"x": x_val,
                                "grid": grid_val},
                          fetch_list=[out],
                          return_numpy=False)
        result = np.array(result)
        path_prefix = "./grid_sampler"
        fluid.io.save_inference_model(path_prefix, ["x", "grid"], [out], exe)
        onnx_path = path_prefix + "/model.onnx"
        program2onnx(
            model_dir=path_prefix,
            save_file=onnx_path,
            opset_version=14,
            enable_onnx_checker=True)

        sess = rt.InferenceSession(onnx_path)
        input_name1 = sess.get_inputs()[0].name
        input_name2 = sess.get_inputs()[1].name
        label_name = sess.get_outputs()[0].name
        pred_onnx = sess.run([label_name],
                             {input_name1: x_data,
                              input_name2: grid_data})[0]
        pred_onnx = np.array(pred_onnx)
        compare(pred_onnx, result, 1e-5, 1e-5)


def test_grid_sample_align_corners_Opset15():
    main_program = paddle.static.Program()
    startup_program = paddle.static.Program()
    # onnxruntime 中的Floor不支持float64
    dtype = 'float32'
    align_corners = False
    N = 5
    with paddle.static.program_guard(main_program, startup_program):
        x = fluid.data(
            name='x', shape=[-1, 6, -1, -1], dtype=dtype, lod_level=1)
        grid = fluid.data(
            name='grid', shape=[-1, 3, -1, -1], dtype=dtype, lod_level=1)
        out = paddle.nn.functional.grid_sample(
            x, grid, align_corners=align_corners)

        exe = paddle.static.Executor(paddle.CPUPlace())
        exe.run(paddle.static.default_startup_program())
        x_data = randtool("int", 1, 10, [N, 6, 3, 3]).astype(dtype)
        grid_data = randtool("float", 1, 10, [N, 3, 4, 2]).astype(dtype)
        x_val = fluid.create_lod_tensor(x_data, [[N]], fluid.CPUPlace())
        grid_val = fluid.create_lod_tensor(grid_data, [[N]], fluid.CPUPlace())
        result, = exe.run(feed={"x": x_val,
                                "grid": grid_val},
                          fetch_list=[out],
                          return_numpy=False)
        result = np.array(result)
        path_prefix = "./grid_sampler"
        fluid.io.save_inference_model(path_prefix, ["x", "grid"], [out], exe)
        onnx_path = path_prefix + "/model.onnx"
        program2onnx(
            model_dir=path_prefix,
            save_file=onnx_path,
            opset_version=15,
            enable_onnx_checker=True)

        sess = rt.InferenceSession(onnx_path)
        input_name1 = sess.get_inputs()[0].name
        input_name2 = sess.get_inputs()[1].name
        label_name = sess.get_outputs()[0].name
        pred_onnx = sess.run([label_name],
                             {input_name1: x_data,
                              input_name2: grid_data})[0]
        pred_onnx = np.array(pred_onnx)
        compare(pred_onnx, result, 1e-5, 1e-5)
