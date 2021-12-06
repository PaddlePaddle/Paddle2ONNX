# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np
import unittest
import abc
import os
import enum
import time
import logging
import shutil
import paddle
import paddle.fluid as fluid
from paddle.fluid.initializer import NumpyArrayInitializer
from paddle.fluid.core import PassVersionChecker
import paddle.fluid.core as core
from paddle import compat as cpt
from typing import Optional, List, Callable, Dict, Any, Set

import hypothesis
from hypothesis import given, settings, seed, reproduce_failure
import hypothesis.strategies as st
from onnxbase import APIOnnx, randtool, compare

logging.basicConfig(level=logging.INFO, format="%(message)s")

settings.register_profile(
    "ci",
    max_examples=100,
    suppress_health_check=hypothesis.HealthCheck.all(),
    deadline=None,
    print_blob=True,
    derandomize=True,
    report_multiple_bugs=False)
settings.register_profile(
    "dev",
    max_examples=1000,
    suppress_health_check=hypothesis.HealthCheck.all(),
    deadline=None,
    print_blob=True,
    derandomize=True,
    report_multiple_bugs=False)
if float(os.getenv('TEST_NUM_PERCENT_CASES', default='1.0')) < 1 or \
    os.getenv('HYPOTHESIS_TEST_PROFILE', 'dev') == 'ci':
    settings.load_profile("ci")
else:
    settings.load_profile("dev")


class AutoScanTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        np.random.seed(1024)
        paddle.enable_static()
        super(AutoScanTest, self).__init__(*args, **kwargs)
        abs_dir = os.path.abspath(os.path.dirname(__file__))
        self.cache_dir = os.path.join(abs_dir,
                                      str(self.__module__) + '_cache_dir')
        self.num_ran_programs = 0
        self.model = None
        self.name = None
        self.test_data_shape = None
        self.input_spec_shape = None
        self.ops = []
        self.opset_version = []

    @abc.abstractmethod
    def sample_convert_configs(self):
        '''
        Generate all config with the combination of different Input tensor shape and
        different Attr values.
        '''
        raise NotImplementedError

    def correct_running_configuration(self, config):
        raise NotImplementedError

    @abc.abstractmethod
    def run_test(self, quant=False):
        raise NotImplementedError

    @abc.abstractmethod
    def ignore_log(self, msg: str):
        logging.warning("SKIP: " + msg)

    @abc.abstractmethod
    def fail_log(self, msg: str):
        logging.error("FAIL: " + msg)

    @abc.abstractmethod
    def success_log(self, msg: str):
        logging.info("SUCCESS: " + msg)


class OPConvertAutoScanTest(AutoScanTest):
    def __init__(self, *args, **kwargs):
        super(OPConvertAutoScanTest, self).__init__(*args, **kwargs)

    def run_and_statis(self,
                       max_examples=100,
                       ops=[],
                       opset_version=[9, 10, 11, 12],
                       reproduce=None,
                       min_success_num=25,
                       max_duration=180):
        if os.getenv('HYPOTHESIS_TEST_PROFILE', 'ci') == "dev":
            max_examples *= 10
            min_success_num *= 10
            # while at ce phase, there's no limit on time
            max_duration = -1
        start_time = time.time()
        settings.register_profile(
            "ci",
            max_examples=max_examples,
            suppress_health_check=hypothesis.HealthCheck.all(),
            deadline=None,
            print_blob=True,
            derandomize=True,
            report_multiple_bugs=False, )
        settings.load_profile("ci")
        logging.info("example: {}".format(max_examples))
        self.ops = ops
        self.opset_version = opset_version

        def sample_convert_generator(draw):
            return self.sample_convert_config(draw)

        def run_test(config):
            return self.run_test(configs=config)

        generator = st.composite(sample_convert_generator)
        loop_func = given(generator())(run_test)
        if reproduce is not None:
            loop_func = reproduce(loop_func)
        logging.info("Start to running test of {}".format(type(self)))

        paddle.disable_static()
        loop_func()

        logging.info(
            "===================Statistical Information===================")
        logging.info("Number of Generated Programs: {}".format(
            self.num_ran_programs))
        successful_ran_programs = int(self.num_ran_programs)
        if successful_ran_programs < min_success_num:
            logging.warning("satisfied_programs = ran_programs")
            logging.error(
                "At least {} programs need to ran successfully, but now only about {} programs satisfied.".
                format(min_success_num, successful_ran_programs))
            assert False
        used_time = time.time() - start_time
        if max_duration > 0 and used_time > max_duration:
            logging.error(
                "The duration exceeds {} seconds, if this is neccessary, try to set a larger number for parameter `max_duration`.".
                format(max_duration))
            assert False

    def run_test(self, configs=None):
        saved_modle = self.model
        saved_opset_version = self.opset_version

        self.correct_running_configuration(configs)

        self.model.eval()
        self.num_ran_programs += 1
        # logging.info("Run model: {}, test_data_shape: {}".format(self.model, self.test_data_shape))
        logging.info("config: {}, test_data_shape: {}".format(
            configs, self.test_data_shape))
        # net, name, ver_list, delta=1e-6, rtol=1e-5
        obj = APIOnnx(self.model, self.name, self.opset_version, self.ops,
                      self.input_spec_shape)

        input_tensors = list()
        name = "input_data"
        for shape in self.test_data_shape:
            temp = paddle.to_tensor(
                randtool("float", -1, 1, shape).astype('float32'))
            input_tensors.append(temp)
        input_tensors = tuple(input_tensors)
        obj.set_input_data(name, input_tensors)

        obj.run()

        self.model = saved_modle
        self.opset_version = saved_opset_version
        logging.info("Run successfully！")
