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

import os
import unittest

import numpy as np
import onnxruntime as ort
from onnxbase import _test_only_pir

import paddle
import paddle2onnx
from paddle.base import unique_name


def convert_scale_to_paddle(onnx_scale, qmax):
    return onnx_scale * qmax / 4


def build_static_net(input_shape, quant_axis, scale_shape, qmin, qmax, type):
    paddle.enable_static()
    main_program = paddle.static.Program()
    startup_program = paddle.static.Program()

    with paddle.static.program_guard(main_program, startup_program):
        x = paddle.static.data(name="x", shape=input_shape, dtype="float32")

        scale = paddle.static.data(name="scale", shape=scale_shape, dtype="float32")
        zero_points = paddle.static.data(
            name="zero_points", shape=scale_shape, dtype="float32"
        )
        accum = paddle.static.data(name="accum", shape=scale_shape, dtype="float32")
        state = paddle.static.data(name="state", shape=scale_shape, dtype="float32")

        x.stop_gradient = True
        dequant_out = paddle.pir.core.create_persistable_value(
            dtype="float32",
            shape=x.shape,
            name=unique_name.generate("dequant_out"),
            initializer=paddle.nn.initializer.Constant(0.0),
            stop_gradient=True,
        )

        dequant_out, _out_state, _out_accum, _out_scale = (
            paddle._C_ops.dequantize_linear(
                x,
                scale,
                zero_points,
                accum,
                state,
                quant_axis,
                8,  # bit_length
                qmin,  # qmin
                qmax,  # qmax
                0,  # rounding_type
                True,  # is_test
                False,  # only_observer
            )
        )

        model_dir = f"./dequantize_linear_model_{type}"
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        paddle.static.save_inference_model(
            os.path.join(model_dir, "model"),
            [x, scale, zero_points, accum, state],
            [dequant_out],
            executor=paddle.static.Executor(paddle.CPUPlace()),
        )

    paddle.disable_static()
    return model_dir


def compare_paddle_and_onnx(
    model_dir, input_data, scale, zero_points, accum, state, opset_version=19
):
    exe = paddle.static.Executor(paddle.CPUPlace())
    paddle.enable_static()
    [inference_program, feed_target_names, fetch_targets] = (
        paddle.static.load_inference_model(os.path.join(model_dir, "model"), exe)
    )
    paddle_result = exe.run(
        inference_program,
        feed={
            feed_target_names[0]: input_data,
            feed_target_names[1]: scale,
            feed_target_names[2]: zero_points,
            feed_target_names[3]: accum,
            feed_target_names[4]: state,
        },
        fetch_list=fetch_targets,
    )[0]
    paddle.disable_static()
    # exit()
    model_file = os.path.join(model_dir, "model.json")
    params_file = ""
    onnx_model_str = paddle2onnx.export(
        model_file,  # model_filename
        params_file,  # params_filename
        None,  # save_file
        opset_version,  # opset_version
        False,  # auto_upgrade_opset
        False,  # dist_prim_all
        True,  # verbose
        True,  # enable_onnx_checker
        True,  # enable_experimental_op
        True,  # enable_optimize
        {},  # custom_ops
        "onnxruntime",  # deploy_backend
        "",  # calibration_file
        "",  # external_file
        False,  # export_fp16_model
    )

    onnx_model_path = os.path.join(model_dir, "model.onnx")
    with open(onnx_model_path, "wb") as f:
        f.write(onnx_model_str)

    sess = ort.InferenceSession(onnx_model_path, providers=["CPUExecutionProvider"])
    input_names = [input.name for input in sess.get_inputs()]
    input_feed = {
        input_names[0]: input_data,
        input_names[1]: scale,
        input_names[2]: zero_points,
        input_names[3]: accum,
        input_names[4]: state,
    }

    onnx_result = sess.run(None, input_feed)[0]

    np.testing.assert_allclose(paddle_result, onnx_result, rtol=1e-5, atol=1e-5)


class TestDequantizeLinear(unittest.TestCase):
    @_test_only_pir
    def test_dequantize_linear_float8_e4m3fn(
        self,
    ):  # except output [0.0, 1.0, 2.0, 896.0, -208.0]
        qmin = -448  # float8_e4m3fn
        qmax = 448  # float8_e4m3fn
        input_data = np.array([0, 0.5, 1, 448, -104]).astype(np.float32)
        scale = np.array([2], dtype=np.float32)
        paddle_scale = convert_scale_to_paddle(scale, qmax)
        zero_points = np.array([0], dtype=np.float32)

        input_shape = input_data.shape
        quant_axis = -1
        scale_shape = [1]
        accum = np.zeros(scale_shape, dtype="float32")
        state = np.zeros(scale_shape, dtype="float32")

        model_dir = build_static_net(
            input_shape, quant_axis, scale_shape, qmin, qmax, "float8_e4m3fn"
        )
        compare_paddle_and_onnx(
            model_dir,
            input_data,
            paddle_scale,
            zero_points,
            accum,
            state,
            opset_version=19,
        )

    @_test_only_pir
    def test_dequantize_linear_float8_e5m2(
        self,
    ):  # except output [0.0, 1.0, 2.0, 98304.0, -192.0]
        qmin = -57344  # float8_e5m2
        qmax = 57344  # float8_e5m2
        input_data = np.array([0, 0.5, 1, 49152, -96]).astype(np.float32)
        scale = np.array([2], dtype=np.float32)
        paddle_scale = convert_scale_to_paddle(scale, qmax)
        zero_points = np.array([0], dtype=np.float32)

        input_shape = input_data.shape
        quant_axis = -1
        scale_shape = [1]
        accum = np.zeros(scale_shape, dtype="float32")
        state = np.zeros(scale_shape, dtype="float32")

        model_dir = build_static_net(
            input_shape, quant_axis, scale_shape, qmin, qmax, "float8_e5m2"
        )
        compare_paddle_and_onnx(
            model_dir,
            input_data,
            paddle_scale,
            zero_points,
            accum,
            state,
            opset_version=19,
        )


if __name__ == "__main__":
    unittest.main()
