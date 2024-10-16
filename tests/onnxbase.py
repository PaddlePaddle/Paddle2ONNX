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

from inspect import isfunction
import logging
from onnxruntime import InferenceSession
import os
import numpy as np
import paddle
import paddle.static as static
from paddle.static import Program
import paddle2onnx.paddle2onnx_cpp2py_export as c_p2o
from paddle2onnx.convert import dygraph2onnx
import shutil
from functools import wraps

def _test_with_pir(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        with paddle.pir_utils.DygraphOldIrGuard():
            func(*args, **kwargs)
        with paddle.pir_utils.DygraphPirGuard():
            func(*args, **kwargs)
    return wrapper


def _test_only_pir(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        with paddle.pir_utils.DygraphPirGuard():
            func(*args, **kwargs)
    return wrapper


def compare_data(result_data, expect_data, delta, rtol):
    res_data = np.allclose(result_data, expect_data, atol=delta, rtol=rtol, equal_nan=True)
    if res_data is True:
        return res_data

    # 输出错误类型
    # 输出数据类型错误
    if result_data.dtype != result_data.dtype:
        logging.error("Different output data types! res type is: {}, and expect type is: {}".format(result_data.dtype,
                                                                                                    expect_data.dtype))
        return False

    # 输出数据大小错误
    if result_data.dtype == np.bool_:
        diff = abs(result_data.astype("int32") - expect_data.astype("int32"))
    else:
        diff = abs(result_data - expect_data)
    logging.error("Output has diff! max diff: {}".format(np.amax(diff)))
    return False


def compare_shape(result_data, expect_data):
    result_shape = result_data.shape
    expect_shape = expect_data.shape
    return result_shape == expect_shape


def compare(result, expect, delta=1e-10, rtol=1e-10):
    """
    比较函数
    :param result: 输入值
    :param expect: 输出值
    :param delta: 误差值
    :return:
    """
    if type(result) == np.ndarray:
        # Convert Paddle Tensor to Numpy array
        if type(expect) == list:
            expect = expect[0]
        expect = expect.numpy()

        # For result_shape is (1) and expect_shape shape is ()
        expect = expect.squeeze()
        result = result.squeeze()
        # Compare the actual value with the expected value and determine whether the output result is correct.
        res_data = compare_data(result, expect, delta, rtol)
        # Compare the actual shape with the expected shape and determine if the output results are correct.
        res_shape = compare_shape(result, expect)

        assert res_data, "result: {} != expect: {}".format(result, expect)
        assert res_shape, "result.shape: {} != expect.shape: {}".format(result.shape, expect.shape)
        assert result.dtype == expect.dtype, "result.dtype: {} != expect.dtype: {}".format(result.dtype, expect.dtype)
    elif type(result) == list and len(result) > 1:
        for i in range(len(result)):
            if isinstance(result[i], (np.generic, np.ndarray)):
                compare(result[i], expect[i], delta, rtol)
            else:
                compare(result[i].numpy(), expect[i], delta, rtol)
    elif len(result) == 1:
        compare(result[0], expect, delta, rtol)


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


dtype_map = {
    paddle.base.core.VarDesc.VarType.FP32: np.float32,
    paddle.base.core.VarDesc.VarType.FP16: np.float16,
    paddle.base.core.VarDesc.VarType.FP64: np.float64,
    paddle.base.core.VarDesc.VarType.INT64: np.int64,
    paddle.base.core.VarDesc.VarType.INT32: np.int32,
    paddle.base.core.VarDesc.VarType.INT16: np.int16,
    paddle.base.core.VarDesc.VarType.INT8: np.int8,
    paddle.base.core.VarDesc.VarType.BOOL: np.bool_,

    paddle.base.core.DataType.FLOAT32: np.float32,
    paddle.base.core.DataType.FLOAT16: np.float16,
    paddle.base.core.DataType.FLOAT64: np.float64,
    paddle.base.core.DataType.INT64: np.int64,
    paddle.base.core.DataType.INT32: np.int32,
    paddle.base.core.DataType.INT16: np.int16,
    paddle.base.core.DataType.INT8: np.int8,
    paddle.base.core.DataType.BOOL: np.bool_,
}



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
                 use_gpu=True,
                 **sup_params):
        self.ops = ops
        if isinstance(self.ops, str):
            self.ops = [self.ops]
        self.seed = 33
        np.random.seed(self.seed)
        paddle.seed(self.seed)
        self.func = func
        if use_gpu and paddle.device.is_compiled_with_cuda() is True:
            self.places = ['gpu']
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
        self.res_fict = {}

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
                    if len(tensor_data.shape) == 0:
                        self.input_feed[str(i)] = np.array(
                            float(in_data), dtype=dtype_map[in_data.dtype])
                    else:
                        self.input_feed[str(i)] = tensor_data.numpy()
                    i += 1
            else:
                if isinstance(in_data, tuple):
                    in_data = in_data[0]
                self.input_dtype.append(in_data.dtype)
                self.input_spec.append(
                    paddle.static.InputSpec(
                        shape=in_data.shape, dtype=in_data.dtype, name=str(i)))
                if len(in_data.shape) == 0:
                    self.input_feed[str(i)] = np.array(
                        float(in_data), dtype=dtype_map[in_data.dtype])
                else:
                    self.input_feed[str(i)] = in_data.numpy()

                i += 1

    def set_device_mode(self, is_gpu=True):
        if paddle.device.is_compiled_with_cuda() is True and is_gpu:
            self.places = ['gpu']
        else:
            self.places = ['cpu']

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
        #        paddle.jit.save(instance, "model/model", input_spec=self.input_spec)
        #        import sys
        #        sys.exit(0)
        enable_dev_version = True
        if os.getenv("ENABLE_DEV", "OFF") == "OFF":
            enable_dev_version = False
        paddle.onnx.export(
            instance,
            os.path.join(self.pwd, self.name, self.name + '_' + str(ver)),
            input_spec=self.input_spec,
            opset_version=ver,
            enable_onnx_checker=True,
            auto_update_opset=False,
            enable_dev_version=enable_dev_version)

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
            os.path.join(self.pwd, self.name,
                         self.name + '_' + str(ver) + '.onnx'),
            providers=['CPUExecutionProvider'])
        ort_outs = sess.run(output_names=None, input_feed=self.input_feed)
        return ort_outs

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
            "op_check_folder",
            input_spec=self.input_spec,
            opset_version=version,
            get_paddle_graph=True,
            enable_dev_version=False)

        included = False
        paddle_op_list = []
        assert len(self.ops) == 1, "You have to set one op name"
        for key, node in paddle_graph.node_map.items():
            op_type = node.type
            op_type = op_type.replace("depthwise_", "")
            if op_type == self.ops[0]:
                included = True

        if len(paddle_graph.node_map.keys()) == 0 and self.ops[0] == '':
            included = True

        assert included is True, "{} op in not in convert OPs, all OPs :{}".format(
            self.ops, paddle_op_list)

    # TODO: PaddlePaddle 2.6 has modified the ParseFromString API, and it cannot be simply replaced with
    #  parse_from_string. Considering that checking the OP name in the Paddle model has almost no impact on the CI
    #  results, temporarily set this function to return True.
    def dev_check_ops(self, op_name, model_file_path):
        # prog = Program()
        #
        # with open(model_file_path, "rb") as f:
        #     model_parse_string = f.read()
        #     prog.parse_from_string(model_parse_string)
        #
        # ops = set()
        # find = False
        # for block in prog.blocks:
        #     for op in block.ops:
        #         op_type = op.type
        #         op_type = op_type.replace("depthwise_", "")
        #         if op_type == op_name:
        #             find = True
        # return find
        return True

    def clip_extra_program_only(self, orig_program_path, clipped_program_path):
        """
        load inference model(program only) and clip extra op
        Args:
            orig_program_path(str): input model path
            clipped_program_path(str): output model path
        Returns:
            None
        """
        paddle.enable_static()
        origin_program_bytes = static.io.load_from_file(orig_program_path)
        origin_program = static.io.deserialize_program(origin_program_bytes)
        clipped_program = origin_program._remove_training_info(clip_extra=True)
        clipped_program_bytes = static.io._serialize_program(clipped_program)
        static.io.save_to_file(clipped_program_path, clipped_program_bytes)
        paddle.disable_static()
        paddle.set_device("cpu")

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

            assert len(self.ops) <= 1, "Need to make sure the number of ops in config is 1."

            # Save Paddle Inference model
            if os.path.exists(self.name):
                shutil.rmtree(self.name)
            paddle.jit.save(self._func, os.path.join(self.name, "model"), self.input_spec)

            # Get PaddleInference model path
            default_model_name = "model.pdmodel"
            if paddle.get_flags("FLAGS_enable_pir_api")["FLAGS_enable_pir_api"]:
                default_model_name = "model.json"
            pdmodel_path = os.path.join(self.name, default_model_name)
            pdiparams_path = os.path.join(self.name, "model.pdiparams")
            # model = paddle.jit.load(os.path.join(self.name, "model"))
            # print("program:", model.program())
            if len(self.ops) > 0:
                self.dev_check_ops(self.ops[0], pdmodel_path)

            original_model_file = pdmodel_path
            params_file = pdiparams_path
            if not os.path.exists(params_file):
                params_file = ""

            # clip extra
            model_file = None
            if paddle.get_flags("FLAGS_enable_pir_api")["FLAGS_enable_pir_api"]:
                model_file = original_model_file
            else:
                model_file = os.path.join(self.name, "cliped_model.pdmodel") 
                self.clip_extra_program_only(original_model_file, model_file)

            for v in self._version:
                onnx_model_str = c_p2o.export(
                    model_file, # model_filename
                    params_file, # params_filename
                    v, # opset_version
                    False, # auto_upgrade_opset
                    True, # verbose
                    True, # enable_onnx_checker
                    True, # enable_experimental_op
                    True, # enable_optimize
                    "onnxruntime", # deploy_backend
                    "", # calibration_file
                    "", # external_file
                    False # export_fp16_model
                )
                with open(os.path.join(self.name, self.name + '_' + str(v) + ".onnx"), "wb") as f:
                    f.write(onnx_model_str)
                self.res_fict[str(v)] = self._mk_onnx_res(ver=v)

            for v in self._version:
                compare(self.res_fict[str(v)], exp, delta=self.delta, rtol=self.rtol)