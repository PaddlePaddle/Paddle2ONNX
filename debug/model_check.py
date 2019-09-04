# Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.
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

import sys
import os
import pickle
import paddle.fluid as fluid
import paddle.fluid.core as core
import numpy as np
from onnx import helper, checker, load
from reader.random_reader import image_classification_random_reader
from reader.image_reader import YoloReader, SSDReader
from debug.onnx_model_helper import split_model, onnx_user_define_fetch_list

TEST = "./tests/test_"
PY = "_op.py"
total_boxes = 0
err_boxes = 0


class Tracker:
    def __init__(self, op_name, op_node):
        self.op_name = op_name
        self.op_node = op_node


def append_fetch_ops(program, fetch_target_names, fetch_holder_name='fetch'):
    """
    In this palce, we will add the fetch op
    """
    global_block = program.global_block()
    fetch_var = global_block.create_var(
        name=fetch_holder_name,
        type=core.VarDesc.VarType.FETCH_LIST,
        persistable=True)
    print("the len of fetch_target_names:%d" % (len(fetch_target_names)))
    for i, name in enumerate(fetch_target_names):

        global_block.append_op(
            type='fetch',
            inputs={'X': [name]},
            outputs={'Out': [fetch_var]},
            attrs={'col': i})


def user_define_fetch_list(program, fetch_target_names, fetch_holder_name):
    """
    In this function, we will remove the old fetch op, add the user define 
    fetch list.
    """
    global_block = program.global_block()
    count = 0
    need_to_remove_op_index = []
    for i, op in enumerate(global_block.ops):
        if op.type == "fetch":
            need_to_remove_op_index.append(i)
    for index in need_to_remove_op_index[::-1]:
        global_block._remove_op(index)
    program.desc.flush()
    # according to fetch list, add the fetch op
    append_fetch_ops(program, fetch_target_names, fetch_holder_name)


def split_onnx_model_to_get_intermedidate(onnx_model, feed_target_names, inputs,
                                          global_block):
    """
    The futncion is according to feed_target_names to compose the small onnx graph model 
    to get intermedidate result.
    """
    onnx_outputs = []
    onnx_runner = Caffe2Backend.prepare(onnx_model, device='CPU')
    layer_outputs = onnx_runner.run(inputs)
    return onnx_outputs


def compare_fluid_onnx_results(fluid_results, onnx_results, feed_target_names,
                               nms_outputs, return_numpy, args):
    """
    Compare the fluid and onnx model layer output data, check decimal of two different models
    """
    global total_boxes
    global err_boxes
    if len(fluid_results) != len(onnx_results):
        raise Exception(
            "The length of fluid_results and onnx_results is not same\
                        ,fluid_results:%d, onnx_results:%d" %
            (len(fluid_results), len(onnx_results)))
    for i in range(0, len(fluid_results)):
        print("start check layer data:%s" % (feed_target_names[i]))
        fluid_result = fluid_results[i]
        if not return_numpy:
            fluid_result = np.array(fluid_result)
        onnx_result = onnx_results[i]
        if feed_target_names[i] in nms_outputs:
            fluid_result = fluid_result[(-fluid_result[:, 1]).argsort()]
            onnx_result = np.squeeze(onnx_result, axis=0)
        for ref, hyp in zip(fluid_result, onnx_result):
            ref = ref.flatten()
            hyp = hyp.flatten()
            total_boxes += 1
            if args.check_task == "image_classification":
                try:
                    np.testing.assert_almost_equal(ref, hyp, decimal=5)
                except:
                    print("layer err {}".format(feed_target_names[i]))
                    break
            else:
                try:
                    np.testing.assert_almost_equal(ref, hyp, decimal=4)
                except:
                    print("layer err {}".format(feed_target_names[i]))
                    print("ref:", ref)
                    print("hyp:", hyp)
                    err_boxes += 1

    return err_boxes / total_boxes


def debug_model(op_list, op_trackers, nms_outputs, args):

    return_numpy = not args.return_variable
    reader = None
    runner_type = "caffe2"
    runner = None
    if args.check_task == "image_classification":
        reader = image_classification_random_reader
    elif args.check_task == "image_detection_yolo":
        detection = YoloReader(args.image_path)
        reader = detection.reader
        runner_type = "onnxruntime"
    elif args.check_task == "image_detection_ssd":
        detection = SSDReader(args.image_path)
        reader = detection.reader
        runner_type = "onnxruntime"
    else:
        raise Exception(
            "Now just support the image_classification and image_detection task")

    feed_var_name = args.name_prefix + "feed"
    fetch_var_name = args.name_prefix + "fetch"
    # start check the op test 
    print("--------------------START CHECK TEST OPS!---------------------")
    for op_name in op_list:
        print("start check the op: %s" % (op_name))
        op_test_name = TEST + op_name + PY
        run_script = "python " + op_test_name
        return_code = os.system(run_script)
        if return_code != 0:
            raise Exception("The op %s test check failed!" % (op_name))
    print("----------------------CHECK TEST OPS OK!-----------------------")

    # In some tools, examples(Tf2Onnx, Caffe2Onnx), therse tools just check the last layer output, 
    # we will check all layers output. Just ensure the robustness of Paddle2Onnx
    # start check the output of op
    print("--------------------START CHECK OPS OUTPUT!--------------------")
    # get the intermediate result of fluid_model & onnx model
    fluid_intermedidate_target_names = []
    op2out = dict()
    out2op = dict()
    for tracker in op_trackers:
        last_node = tracker.op_node[0]
        outputs = last_node.output
        op2out[tracker] = outputs
        for output in outputs:
            out2op[output] = tracker
        fluid_intermedidate_target_names.extend(outputs)

    #fluid_intermedidate_target_names = fluid_intermedidate_target_names[:]
    # load the paddle and onnx model 
    # init the fluid executor 

    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    feed_target_names = None
    if len(args.fluid_model_name) != 0 and len(args.fluid_params_name) != 0:
        [fluid_infer_program, feed_target_names,
         fetch_targets] = fluid.io.load_inference_model(args.fluid_model, exe,
                                                        args.fluid_model_name,
                                                        args.fluid_params_name)
    else:
        [fluid_infer_program, feed_target_names,
         fetch_targets] = fluid.io.load_inference_model(args.fluid_model, exe)
    fetch_target_names = [target.name for target in fetch_targets]
    fluid_intermedidate_target_names = []
    fluid_intermedidate_target_names.extend(fetch_target_names)
    # in this section, wo will set the varaiable we want to get 
    global_block = fluid_infer_program.global_block()

    fetch_list = [global_block.var(name) for name in fluid_intermedidate_target_names\
                 if global_block.has_var(name)]

    fluid_intermedidate_target_names = [var.name for var in fetch_list]

    # load the onnx model and init the onnx executor  
    onnx_model = load(args.onnx_model)
    # user define the fetch list 
    onnx_model = onnx_user_define_fetch_list(onnx_model, global_block,
                                             fluid_intermedidate_target_names)
    user_define_fetch_list(fluid_infer_program,
                           fluid_intermedidate_target_names, fetch_var_name)
    if runner_type == "caffe2":
        from caffe2.python.onnx.backend import Caffe2Backend
        runner = Caffe2Backend.prepare(onnx_model, device='CPU')
    for inputs in reader(fluid_infer_program,\
        feed_target_names):
        fluid_results = exe.run(fluid_infer_program,
                                feed=dict(zip(feed_target_names, inputs)),
                                fetch_list=fetch_list,
                                feed_var_name=feed_var_name,
                                fetch_var_name=fetch_var_name,
                                return_numpy=return_numpy)
        print("the fluid results len:%d" % (len(fluid_results)))
        if runner == None:
            #save model to tests dir and run python script
            with open("tests/nms_test.onnx", 'wb') as f:
                f.write(onnx_model.SerializeToString())
            f.close()
            with open("tests/inputs_test.pkl", 'wb') as f:
                pickle.dump(dict(zip(feed_target_names, inputs)), f)
            f.close()
            ret = os.system("python tests/onnx_runtime.py %s %s %s" %
                            (False, False, False))
            with open("tests/outputs_test.pkl", "rb") as f:
                onnx_results = pickle.load(f)
            f.close()
        else:
            onnx_results = runner.run(inputs)
        print("the onnx_results len:%d" % (len(onnx_results)))
        err_ratio = compare_fluid_onnx_results(fluid_results, onnx_results,
                                               fluid_intermedidate_target_names,
                                               nms_outputs, return_numpy, args)
    if err_ratio < 0.01:
        raise Exception("The result between onnx and paddle has difference")
