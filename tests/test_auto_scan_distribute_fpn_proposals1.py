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

import paddle
from paddle2onnx.command import program2onnx
import onnxruntime as rt

paddle.enable_static()

import numpy as np
import paddle
import paddle.fluid as fluid
from onnxbase import randtool, compare


def test_generate_proposals():
    main_program = paddle.static.Program()
    startup_program = paddle.static.Program()

    with paddle.static.program_guard(main_program, startup_program):
        fpn_rois = fluid.data(
            name='fpn_rois', shape=[-1, 4], dtype='float32', lod_level=1)

        def init_test_input():
            images_shape = [512, 512]
            rois_lod = [[100, 200]]
            rois = []
            lod = rois_lod[0]
            bno = 0
            for roi_num in lod:
                for i in range(roi_num):
                    xywh = np.random.rand(4)
                    xy1 = xywh[0:2] * 20
                    wh = xywh[2:4] * (images_shape - xy1)
                    xy2 = xy1 + wh
                    roi = [bno, xy1[0], xy1[1], xy2[0], xy2[1]]
                    rois.append(roi)
                bno += 1
            rois = np.array(rois).astype("float32")
            rois = rois[:, 1:5]
            return rois, [rois.shape[0]]

        fpn_rois_data, rois_num_data = init_test_input()

        min_level = 2
        max_level = 5
        refer_level = 4
        refer_scale = 224

        out = fluid.layers.distribute_fpn_proposals(
            fpn_rois, min_level, max_level, refer_level, refer_scale)
        exe = paddle.static.Executor(paddle.CPUPlace())
        exe.run(paddle.static.default_startup_program())
        fpn_rois_val = fluid.create_lod_tensor(fpn_rois_data,
                                               [[fpn_rois_data.shape[0]]],
                                               fluid.CPUPlace())
        result = exe.run(feed={"fpn_rois": fpn_rois_val},
                         fetch_list=list(out),
                         return_numpy=False)
        path_prefix = "./distribute_fpn_proposals"
        fluid.io.save_inference_model(
            path_prefix, ["fpn_rois"],
            [out[0][0], out[0][1], out[0][2], out[0][3], out[1]], exe)

        onnx_path = path_prefix + "/model.onnx"
        program2onnx(
            model_dir=path_prefix,
            save_file=onnx_path,
            opset_version=11,
            enable_onnx_checker=True)

        sess = rt.InferenceSession(
            onnx_path, providers=['CPUExecutionProvider'])
        input_name1 = sess.get_inputs()[0].name
        pred_onnx = sess.run(None, {input_name1: fpn_rois_data})

        compare(pred_onnx, result, 1e-5, 1e-5)


def test_generate_proposalsOpWithRoisNum():
    main_program = paddle.static.Program()
    startup_program = paddle.static.Program()

    with paddle.static.program_guard(main_program, startup_program):
        fpn_rois = fluid.layers.data(
            name='fpn_rois',
            shape=[-1, 4],
            append_batch_size=False,
            dtype='float32')
        rois_num = fluid.layers.data(
            name='rois_num', shape=[1], append_batch_size=False, dtype='int32')

        def init_test_input():
            images_shape = [512, 512]
            rois_lod = [[100, 200]]
            rois = []
            lod = rois_lod[0]
            bno = 0
            for roi_num in lod:
                for i in range(roi_num):
                    xywh = np.random.rand(4)
                    xy1 = xywh[0:2] * 20
                    wh = xywh[2:4] * (images_shape - xy1)
                    xy2 = xy1 + wh
                    roi = [bno, xy1[0], xy1[1], xy2[0], xy2[1]]
                    rois.append(roi)
                bno += 1
            rois = np.array(rois).astype("float32")
            rois = rois[:, 1:5]
            return rois, [rois.shape[0]]

        fpn_rois_data, rois_num_data = init_test_input()

        min_level = 2
        max_level = 5
        refer_scale = 224
        refer_level = 4

        out = fluid.layers.distribute_fpn_proposals(
            fpn_rois, min_level, max_level, refer_level, refer_scale, rois_num)
        exe = paddle.static.Executor(paddle.CPUPlace())
        exe.run(paddle.static.default_startup_program())
        result = exe.run(
            feed={"fpn_rois": fpn_rois_data,
                  "rois_num": rois_num_data},
            fetch_list=list(out),
            return_numpy=False)
        pass
        path_prefix = "./distribute_fpn_proposals1"
        fluid.io.save_inference_model(path_prefix, ["fpn_rois", "rois_num"], [
            out[0][0], out[0][1], out[0][2], out[0][3], out[1], out[2][0],
            out[2][1], out[2][2], out[2][3]
        ], exe)

        onnx_path = path_prefix + "/model.onnx"
        program2onnx(
            model_dir=path_prefix,
            save_file=onnx_path,
            opset_version=12,
            enable_onnx_checker=True)

        sess = rt.InferenceSession(
            onnx_path, providers=['CPUExecutionProvider'])
        input_name1 = sess.get_inputs()[0].name
        input_name2 = sess.get_inputs()[1].name
        pred_onnx = sess.run(None, {
            input_name1: fpn_rois_data,
            input_name2: rois_num_data,
        })
        compare(pred_onnx, result, 1e-5, 1e-5)


if __name__ == "__main__":
    test_generate_proposals()
    test_generate_proposalsOpWithRoisNum()
